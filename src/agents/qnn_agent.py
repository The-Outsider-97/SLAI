"""SLAI QNN agent backed by a bounded NumPy state-vector simulator.

The QNN agent is the orchestration boundary for SLAI's parameterized
state-vector capability. Numerical ingress, Born measurement / objective
mathematics, and resource / tuning policy are delegated to ``src.agents.qnn``.

Important
---------
This implementation is a classical state-vector simulation of a variational
quantum circuit. It does not claim quantum speed-up, quantum-hardware execution,
or a recurrent quantum architecture.

Dependency direction
--------------------
The agent intentionally does not import ``qnn.quantum_memory`` or
``qnn.utils.config_loader``. Durable checkpointing and tuning remain external
SLAI services reached through explicit state / mutation hooks rather than
reverse imports into their orchestration layers.
"""

from __future__ import annotations

__version__ = "2.2.0"

import copy
import hashlib
import json
import math
import time
import numpy as np


from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from .base.utils.main_config_loader import get_config_section
from .base_agent import BaseAgent
from .qnn.quantum_encoding import *
from .qnn.quantum_mno import *
from .qnn.quantum_policy import *
from .qnn.utils.quantum_errors import *
from .qnn.utils.quantum_helpers import *
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("Quantum Neural Network Agent")
printer = PrettyPrinter()


_CHECKPOINT_SCHEMA = "slai.qnn-agent.state.v1"

# Current quantum_policy defaults. These values mirror
# src/agents/qnn/configs/quantum_config.yaml and are checked against the
# policy-owned configuration at initialization so configuration drift fails
# visibly rather than silently changing the agent's resource envelope.
_POLICY_DEFAULTS = {
    "max_statevector_bytes": 16_777_216,
    "max_parameter_bytes": 16_777_216,
    "max_working_set_bytes": 268_435_456,
    "max_sequence_length": 128,
    "max_tasks_per_request": 32,
    "max_gradient_evaluations": 1_000_000,
    "max_training_steps": 1_000,
}

_SUPPORTED_ENTANGLEMENT = frozenset({"none", "linear", "ring"})
_SUPPORTED_LOSSES = frozenset({"state_fidelity", "probability_mse"})
_SUPPORTED_GRADIENT_METHODS = frozenset({"parameter_shift", "finite_difference"})


# ---------------------------------------------------------------------------
# Local circuit / request contracts
# ---------------------------------------------------------------------------
#
# These contracts currently live here because SLAI-v.2.2 does not contain a
# qnn module that exports QNNConfig, QNNTask, StateVectorCircuit, or gate
# primitives. Keeping the implementation explicit in this file makes the
# facade executable today without pretending those symbols exist elsewhere.
# The numerical boundary remains small and has no persistence / tuning search
# ownership, so it can later be extracted mechanically to qnn/quantum_circuit.py
# and qnn/quantum_types.py without changing QNNAgent's public API.


def _finite_scalar(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise QNNConfigurationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise QNNConfigurationError(f"{name} must be finite")
    return result


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise QNNConfigurationError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise QNNConfigurationError(f"{name} must be non-negative")
    return result


def _normalized_choice(
    value: Any,
    *,
    name: str,
    allowed: frozenset[str],
    aliases: Mapping[str, str] | None = None,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QNNConfigurationError(f"{name} must be a non-empty string")
    normalized = value.strip().casefold()
    if aliases:
        normalized = aliases.get(normalized, normalized)
    if normalized not in allowed:
        supported = ", ".join(sorted(allowed))
        raise QNNConfigurationError(f"{name} must be one of: {supported}")
    return normalized


@dataclass(frozen=True, slots=True)
class QNNConfig:
    """Validated QNN architecture, optimizer, and resource envelope.

    Defaults preserve the historical v2.2 architecture where evidence exists
    (four qubits, two layers, 0.01 learning rate, adjacent/linear
    entanglement). Numerical differentiation / clipping defaults are
    implementation safeguards, not empirically tuned claims.
    """

    num_qubits: int = 4
    num_quantum_layers: int = 2
    learning_rate: float = 0.01
    seed: int = 42
    entanglement: str = "linear"
    loss: str = "state_fidelity"
    gradient_method: str = "parameter_shift"
    finite_difference_step: float = 1.0e-5
    gradient_clip_norm: float = 1.0

    max_statevector_bytes: int = _POLICY_DEFAULTS["max_statevector_bytes"]
    max_parameter_bytes: int = _POLICY_DEFAULTS["max_parameter_bytes"]
    max_working_set_bytes: int = _POLICY_DEFAULTS["max_working_set_bytes"]
    max_sequence_length: int = _POLICY_DEFAULTS["max_sequence_length"]
    max_tasks_per_request: int = _POLICY_DEFAULTS["max_tasks_per_request"]
    max_gradient_evaluations: int = _POLICY_DEFAULTS["max_gradient_evaluations"]
    max_training_steps: int = _POLICY_DEFAULTS["max_training_steps"]

    _PUBLIC_FIELDS = frozenset(
        {
            "num_qubits",
            "num_quantum_layers",
            "learning_rate",
            "seed",
            "entanglement",
            "loss",
            "gradient_method",
            "finite_difference_step",
            "gradient_clip_norm",
            "max_statevector_bytes",
            "max_parameter_bytes",
            "max_working_set_bytes",
            "max_sequence_length",
            "max_tasks_per_request",
            "max_gradient_evaluations",
            "max_training_steps",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "num_qubits", positive_int(self.num_qubits, "num_qubits"))
        object.__setattr__(
            self,
            "num_quantum_layers",
            positive_int(self.num_quantum_layers, "num_quantum_layers"),
        )
        object.__setattr__(
            self,
            "learning_rate",
            positive_float(self.learning_rate, "learning_rate"),
        )
        object.__setattr__(self, "seed", _non_negative_int(self.seed, "seed"))
        object.__setattr__(
            self,
            "entanglement",
            _normalized_choice(
                self.entanglement,
                name="entanglement",
                allowed=_SUPPORTED_ENTANGLEMENT,
                aliases={"adjacent": "linear"},
            ),
        )
        object.__setattr__(
            self,
            "loss",
            _normalized_choice(
                self.loss,
                name="loss",
                allowed=_SUPPORTED_LOSSES,
                aliases={"fidelity": "state_fidelity", "mse": "probability_mse"},
            ),
        )
        object.__setattr__(
            self,
            "gradient_method",
            _normalized_choice(
                self.gradient_method,
                name="gradient_method",
                allowed=_SUPPORTED_GRADIENT_METHODS,
            ),
        )
        object.__setattr__(
            self,
            "finite_difference_step",
            positive_float(
                self.finite_difference_step,
                "finite_difference_step",
            ),
        )
        object.__setattr__(
            self,
            "gradient_clip_norm",
            positive_float(self.gradient_clip_norm, "gradient_clip_norm"),
        )

        for name in (
            "max_statevector_bytes",
            "max_parameter_bytes",
            "max_working_set_bytes",
            "max_sequence_length",
            "max_tasks_per_request",
            "max_gradient_evaluations",
            "max_training_steps",
        ):
            object.__setattr__(self, name, positive_int(getattr(self, name), name))

        if self.gradient_method == "parameter_shift" and self.loss != "state_fidelity":
            raise QNNConfigurationError(
                "parameter_shift is supported only with state_fidelity loss"
            )

        if self.statevector_bytes > self.max_statevector_bytes:
            raise QNNResourceLimitError(
                "configured QNN state vector exceeds max_statevector_bytes: "
                f"required={self.statevector_bytes}, "
                f"limit={self.max_statevector_bytes}"
            )
        if self.parameter_bytes > self.max_parameter_bytes:
            raise QNNResourceLimitError(
                "configured QNN parameters exceed max_parameter_bytes: "
                f"required={self.parameter_bytes}, "
                f"limit={self.max_parameter_bytes}"
            )
        conservative_minimum = 4 * self.statevector_bytes + 4 * self.parameter_bytes
        if conservative_minimum > self.max_working_set_bytes:
            raise QNNResourceLimitError(
                "configured QNN cannot fit one conservative operation in "
                "max_working_set_bytes: "
                f"required={conservative_minimum}, "
                f"limit={self.max_working_set_bytes}"
            )

    @property
    def state_dimension(self) -> int:
        return 1 << self.num_qubits

    @property
    def statevector_bytes(self) -> int:
        return self.state_dimension * np.dtype(np.complex128).itemsize

    @property
    def parameter_count(self) -> int:
        return self.num_quantum_layers * self.num_qubits * 3

    @property
    def parameter_bytes(self) -> int:
        return self.parameter_count * np.dtype(np.float64).itemsize

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "QNNConfig":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise QNNConfigurationError("QNN configuration must be a mapping")
        raw = dict(value)
        unknown = set(raw) - cls._PUBLIC_FIELDS
        if unknown:
            raise QNNConfigurationError(
                "QNN configuration contains unsupported field(s): "
                f"{', '.join(sorted(str(item) for item in unknown))}"
            )
        try:
            return cls(**raw)
        except TypeError as exc:
            raise QNNConfigurationError(
                f"unable to construct QNN configuration: {exc}"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in sorted(self._PUBLIC_FIELDS)}


@dataclass(frozen=True, slots=True)
class QNNTask:
    """Normalized task contract for inference, evaluation, or training."""

    input_sequences: tuple[Any, ...]
    target_outputs: tuple[Any, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        inputs = self._normalize_sequence(self.input_sequences, "input_sequences")
        object.__setattr__(self, "input_sequences", inputs)

        targets = self.target_outputs
        if targets is not None:
            normalized_targets = self._normalize_sequence(targets, "target_outputs")
            if len(normalized_targets) != len(inputs):
                raise QNNInputError(
                    "target_outputs must contain the same number of states "
                    "as input_sequences"
                )
            object.__setattr__(self, "target_outputs", normalized_targets)

        if not isinstance(self.metadata, Mapping):
            raise QNNInputError("task metadata must be a mapping")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @staticmethod
    def _normalize_sequence(value: Any, name: str) -> tuple[Any, ...]:
        if isinstance(value, np.ndarray):
            if value.ndim == 1:
                return (value,)
            if value.ndim == 2:
                values = tuple(value[index] for index in range(value.shape[0]))
            else:
                raise QNNInputError(
                    f"{name} must be a one- or two-dimensional NumPy array"
                )
        else:
            if isinstance(value, (str, bytes, bytearray)) or not isinstance(
                value, Sequence
            ):
                raise QNNInputError(f"{name} must be a sequence of state vectors")
            values = tuple(value)
            if values and all(np.isscalar(item) for item in values):
                values = (values,)
        if not values:
            raise QNNInputError(f"{name} must not be empty")
        return values

    @classmethod
    def from_value(cls, value: "QNNTask | Mapping[str, Any] | Any") -> "QNNTask":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            if "input_sequences" not in value:
                raise QNNInputError("QNN task requires input_sequences")
            return cls(
                input_sequences=value["input_sequences"],
                target_outputs=value.get("target_outputs"),
                metadata=dict(value.get("metadata") or {}),
            )

        if not hasattr(value, "input_sequences"):
            raise QNNInputError(
                "QNN task must be QNNTask, a mapping, or expose input_sequences"
            )
        return cls(
            input_sequences=getattr(value, "input_sequences"),
            target_outputs=getattr(value, "target_outputs", None),
            metadata=dict(getattr(value, "metadata", {}) or {}),
        )


QuantumGate: TypeAlias = np.ndarray

HADAMARD = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0)
PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
PAULI_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
CNOT = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    dtype=np.complex128,
)


def rx(theta: float) -> np.ndarray:
    angle = _finite_scalar(theta, "theta")
    c = math.cos(angle / 2.0)
    s = math.sin(angle / 2.0)
    return np.array([[c, -1.0j * s], [-1.0j * s, c]], dtype=np.complex128)


def ry(theta: float) -> np.ndarray:
    angle = _finite_scalar(theta, "theta")
    c = math.cos(angle / 2.0)
    s = math.sin(angle / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz(theta: float) -> np.ndarray:
    angle = _finite_scalar(theta, "theta")
    return np.array(
        [
            [np.exp(-0.5j * angle), 0.0],
            [0.0, np.exp(0.5j * angle)],
        ],
        dtype=np.complex128,
    )


def zero_state(num_qubits: int) -> np.ndarray:
    qubits = positive_int(num_qubits, "num_qubits")
    state = np.zeros(1 << qubits, dtype=np.complex128)
    state[0] = 1.0 + 0.0j
    return state


def apply_gate(
    state: Any,
    gate: Any,
    target_qubits: Sequence[int],
    *,
    num_qubits: int,
) -> np.ndarray:
    """Apply a k-qubit gate without materializing a full-system matrix."""

    qubits = positive_int(num_qubits, "num_qubits")
    expected_dimension = 1 << qubits

    try:
        vector = np.asarray(state, dtype=np.complex128)
    except (TypeError, ValueError) as exc:
        raise QNNInputError("state cannot be converted to complex128") from exc
    if vector.shape != (expected_dimension,):
        raise QNNInputError(
            f"state must have shape ({expected_dimension},), got {vector.shape}"
        )
    if not np.all(np.isfinite(vector.real)) or not np.all(np.isfinite(vector.imag)):
        raise QNNInputError("state contains non-finite amplitudes")

    if isinstance(target_qubits, (str, bytes, bytearray)) or not isinstance(
        target_qubits, Sequence
    ):
        raise QNNInputError("target_qubits must be a sequence")
    targets = tuple(target_qubits)
    if not targets:
        raise QNNInputError("target_qubits must not be empty")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in targets):
        raise QNNInputError("target_qubits must contain integers")
    if len(set(targets)) != len(targets):
        raise QNNInputError("target_qubits must not contain duplicates")
    if any(item < 0 or item >= qubits for item in targets):
        raise QNNInputError("target_qubits contains an out-of-range qubit")

    width = 1 << len(targets)
    try:
        matrix = np.asarray(gate, dtype=np.complex128)
    except (TypeError, ValueError) as exc:
        raise QNNInputError("gate cannot be converted to complex128") from exc
    if matrix.shape != (width, width):
        raise QNNInputError(
            f"gate for {len(targets)} qubit(s) must have shape "
            f"({width}, {width}), got {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix.real)) or not np.all(np.isfinite(matrix.imag)):
        raise QNNInputError("gate contains non-finite values")

    # Axis 0 corresponds to qubit 0. ``targets`` order determines the basis
    # ordering expected by the local gate matrix.
    tensor = vector.reshape((2,) * qubits)
    remaining = tuple(index for index in range(qubits) if index not in targets)
    permutation = targets + remaining
    inverse = np.argsort(permutation)
    front = np.transpose(tensor, permutation).reshape(width, -1)
    updated = matrix @ front
    result = np.transpose(
        updated.reshape((2,) * qubits),
        inverse,
    ).reshape(expected_dimension)

    if not np.all(np.isfinite(result.real)) or not np.all(np.isfinite(result.imag)):
        raise QNNInputError("gate application produced non-finite amplitudes")
    return np.ascontiguousarray(result, dtype=np.complex128)


class StateVectorCircuit:
    """Parameterized Rx-Ry-Rz circuit with fixed CNOT entanglers."""

    def __init__(self, *, num_qubits: int, num_layers: int, entanglement: str) -> None:
        self.num_qubits = positive_int(num_qubits, "num_qubits")
        self.num_layers = positive_int(num_layers, "num_layers")
        self.entanglement = _normalized_choice(
            entanglement,
            name="entanglement",
            allowed=_SUPPORTED_ENTANGLEMENT,
            aliases={"adjacent": "linear"},
        )
        self.state_dimension = 1 << self.num_qubits
        self.weight_shape = (self.num_layers, self.num_qubits, 3)

    def validate_weights(self, value: Any) -> np.ndarray:
        try:
            raw = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise QNNInputError("quantum weights cannot be converted to an array") from exc
        if np.iscomplexobj(raw):
            raise QNNInputError("quantum weights must be real-valued")
        try:
            weights = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise QNNInputError("quantum weights must be float-compatible") from exc
        if weights.shape != self.weight_shape:
            raise QNNInputError(
                f"quantum weights must have shape {self.weight_shape}, "
                f"got {weights.shape}"
            )
        if not np.all(np.isfinite(weights)):
            raise QNNInputError("quantum weights contain non-finite values")
        return np.ascontiguousarray(weights, dtype=np.float64)

    def _entanglement_pairs(self) -> tuple[tuple[int, int], ...]:
        if self.entanglement == "none" or self.num_qubits < 2:
            return ()
        pairs = [(index, index + 1) for index in range(self.num_qubits - 1)]
        if self.entanglement == "ring" and self.num_qubits > 2:
            pairs.append((self.num_qubits - 1, 0))
        return tuple(pairs)

    def apply_layer(self, state: Any, layer_weights: Any) -> np.ndarray:
        try:
            weights = np.asarray(layer_weights, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise QNNInputError("layer weights must be float-compatible") from exc
        expected = (self.num_qubits, 3)
        if weights.shape != expected or not np.all(np.isfinite(weights)):
            raise QNNInputError(
                f"layer weights must be finite with shape {expected}"
            )

        current = np.asarray(state, dtype=np.complex128)
        if current.shape != (self.state_dimension,):
            raise QNNInputError(
                f"state must have shape ({self.state_dimension},)"
            )

        for qubit in range(self.num_qubits):
            current = apply_gate(
                current,
                rx(float(weights[qubit, 0])),
                (qubit,),
                num_qubits=self.num_qubits,
            )
            current = apply_gate(
                current,
                ry(float(weights[qubit, 1])),
                (qubit,),
                num_qubits=self.num_qubits,
            )
            current = apply_gate(
                current,
                rz(float(weights[qubit, 2])),
                (qubit,),
                num_qubits=self.num_qubits,
            )

        for control, target in self._entanglement_pairs():
            current = apply_gate(
                current,
                CNOT,
                (control, target),
                num_qubits=self.num_qubits,
            )

        return np.ascontiguousarray(current, dtype=np.complex128)

    def forward(self, state: Any, weights: Any) -> np.ndarray:
        parameters = self.validate_weights(weights)
        current = np.asarray(state, dtype=np.complex128)
        if current.shape != (self.state_dimension,):
            raise QNNInputError(
                f"state must have shape ({self.state_dimension},)"
            )
        for layer_index in range(self.num_layers):
            current = self.apply_layer(current, parameters[layer_index])
        return np.ascontiguousarray(current, dtype=np.complex128)


class QNNAgent(BaseAgent):
    """Parameterized state-vector QNN with explicit SLAI runtime boundaries."""

    capabilities = (
        "quantum_state_simulation",
        "quantum_circuit_inference",
        "quantum_circuit_training",
        "quantum_state_evaluation",
    )

    _MODEL_STATE_FIELDS = frozenset({"quantum_weights", "training_step"})
    _MODEL_CONFIG_FIELDS = QNNConfig._PUBLIC_FIELDS
    _AGENT_CONFIG_FIELDS = _MODEL_CONFIG_FIELDS | {"shared_memory"}

    _DEFAULT_SHARED_MEMORY_CONFIG = {
        "enabled": True,
        "ttl_seconds": None,
        "publish_notifications": False,
        "summary_key_prefix": "qnn:summary",
        "event_channel": "qnn.events",
    }
    CHECKPOINTING_SUPPORTED = True

    def __init__(
        self,
        shared_memory: Any = None,
        agent_factory: Any = None,
        config: Mapping[str, Any] | None = None,
        *,
        checkpoint_manager: Any = None,
    ) -> None:
        if config is not None and not isinstance(config, Mapping):
            raise QNNConfigurationError("QNN config override must be a mapping")

        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            config=config,
            checkpoint_manager=checkpoint_manager,
        )

        resolved = dict(
            get_config_section(
                "qnn_agent",
                config=self.config,
            )
            or {}
        )
        if config:
            resolved.update(dict(config))

        unknown = set(resolved) - self._AGENT_CONFIG_FIELDS
        if unknown:
            raise QNNConfigurationError(
                "qnn_agent configuration contains unsupported field(s): "
                f"{', '.join(sorted(str(item) for item in unknown))}"
            )

        model_config = {
            key: value for key, value in resolved.items() if key in self._MODEL_CONFIG_FIELDS
        }
        self.qnn_config = QNNConfig.from_mapping(model_config)
        self._shared_memory_config = self._resolve_shared_memory_config(
            resolved.get("shared_memory")
        )

        self._rng = np.random.default_rng(self.qnn_config.seed)
        self._circuit = self._build_circuit(self.qnn_config)
        self._encoder = self._build_encoder(self.qnn_config)
        self._policy = self._build_policy(self.qnn_config)
        self.measurement_optimizer = self._build_measurement_optimizer(
            self.qnn_config
        )
        self.qnn_metrics = self.measurement_optimizer.metrics
        self.evaluator = self.qnn_metrics

        self.quantum_weights = self._initialize_quantum_weights()
        self._training_step = 0
        self._last_gradient: np.ndarray | None = None
        self.gate_definitions = self._define_quantum_gates()

        logger.info(
            "QNN Agent initialized | qubits=%d | layers=%d | entanglement=%s | "
            "loss=%s | gradient=%s",
            self.num_qubits,
            self.num_quantum_layers,
            self.entanglement,
            self.loss,
            self.gradient_method,
        )

    # ------------------------------------------------------------------
    # Construction and configuration
    # ------------------------------------------------------------------
    @staticmethod
    def _build_circuit(config: QNNConfig) -> StateVectorCircuit:
        return StateVectorCircuit(
            num_qubits=config.num_qubits,
            num_layers=config.num_quantum_layers,
            entanglement=config.entanglement,
        )

    @staticmethod
    def _build_encoder(config: QNNConfig) -> StateVectorEncoder:
        return StateVectorEncoder(state_dimension=config.state_dimension)

    @staticmethod
    def _build_policy(config: QNNConfig) -> QuantumExecutionPolicy:
        policy = QuantumExecutionPolicy.from_config(config)

        # ``QuantumExecutionPolicy`` currently receives resource values from
        # QNNConfig while also owning the same limits in quantum_config.yaml.
        # Reject drift explicitly until that duplication is removed from the
        # subsystem API.
        configured = policy.policy_config
        for name in _POLICY_DEFAULTS:
            if name not in configured:
                raise QNNConfigurationError(
                    f"quantum_policy.{name} is required"
                )
            owned_value = positive_int(
                configured[name],
                f"quantum_policy.{name}",
            )
            if name in {"max_statevector_bytes", "max_parameter_bytes"}:
                runtime_value = int(getattr(config, name))
            else:
                runtime_value = int(getattr(policy, name))
            if runtime_value != owned_value:
                raise QNNConfigurationError(
                    f"qnn_agent.{name}={runtime_value} disagrees with "
                    f"quantum_policy.{name}={owned_value}; resource policy must "
                    "have one source of truth"
                )
        return policy

    @staticmethod
    def _build_measurement_optimizer(
        config: QNNConfig,
    ) -> QuantumMeasurementOptimizer:
        return QuantumMeasurementOptimizer(
            loss=config.loss,
            gradient_method=config.gradient_method,
            finite_difference_step=config.finite_difference_step,
            gradient_clip_norm=config.gradient_clip_norm,
        )

    @classmethod
    def _resolve_shared_memory_config(cls, value: Any) -> dict[str, Any]:
        if value is None:
            section: dict[str, Any] = {}
        elif isinstance(value, Mapping):
            section = dict(value)
        else:
            raise QNNConfigurationError("qnn_agent.shared_memory must be a mapping")

        unknown = set(section) - set(cls._DEFAULT_SHARED_MEMORY_CONFIG)
        if unknown:
            raise QNNConfigurationError(
                "qnn_agent.shared_memory contains unsupported field(s): "
                f"{', '.join(sorted(str(item) for item in unknown))}"
            )

        resolved = {**cls._DEFAULT_SHARED_MEMORY_CONFIG, **section}
        for name in ("enabled", "publish_notifications"):
            if not isinstance(resolved[name], bool):
                raise QNNConfigurationError(
                    f"qnn_agent.shared_memory.{name} must be a boolean"
                )

        ttl = resolved["ttl_seconds"]
        if ttl is not None:
            ttl = _finite_scalar(ttl, "qnn_agent.shared_memory.ttl_seconds")
            if ttl <= 0.0:
                raise QNNConfigurationError(
                    "qnn_agent.shared_memory.ttl_seconds must be positive"
                )
        resolved["ttl_seconds"] = ttl

        for name in ("summary_key_prefix", "event_channel"):
            item = resolved[name]
            if not isinstance(item, str) or not item.strip():
                raise QNNConfigurationError(
                    f"qnn_agent.shared_memory.{name} must be a non-empty string"
                )
            resolved[name] = item.strip()
        return resolved

    @property
    def num_qubits(self) -> int:
        return self.qnn_config.num_qubits

    @property
    def num_quantum_layers(self) -> int:
        return self.qnn_config.num_quantum_layers

    @property
    def learning_rate(self) -> float:
        return self.qnn_config.learning_rate

    @property
    def seed(self) -> int:
        return self.qnn_config.seed

    @property
    def entanglement(self) -> str:
        return self.qnn_config.entanglement

    @property
    def loss(self) -> str:
        return self.measurement_optimizer.loss

    @property
    def gradient_method(self) -> str:
        return self.measurement_optimizer.gradient_method

    @property
    def finite_difference_step(self) -> float:
        return self.measurement_optimizer.finite_difference_step

    @property
    def gradient_clip_norm(self) -> float:
        return self.measurement_optimizer.gradient_clip_norm

    @property
    def estimated_statevector_bytes(self) -> int:
        return self.qnn_config.statevector_bytes

    @property
    def parameter_bytes(self) -> int:
        return self.qnn_config.parameter_bytes

    @property
    def training_step(self) -> int:
        return self._training_step

    # ------------------------------------------------------------------
    # State preparation and resource admission
    # ------------------------------------------------------------------
    def _initialize_quantum_weights(self) -> np.ndarray:
        return self._rng.uniform(
            low=-np.pi,
            high=np.pi,
            size=self._circuit.weight_shape,
        ).astype(np.float64)

    def _normalize_sequence(
        self,
        sequence: Sequence[Any],
        *,
        name: str,
    ) -> tuple[np.ndarray, ...]:
        encoded = self._encoder.encode_sequence(sequence, name=name)
        self._policy.validate_sequence_length(len(encoded), name=name)
        return encoded

    def _validate_working_set(self, state_slots: int, *, operation: str) -> None:
        self._policy.validate_working_set(state_slots, operation=operation)

    def apply_tuning_parameters(self, parameters: Mapping[str, Any]) -> dict[str, Any]:
        """Apply declared QNN tunables atomically.

        This is deliberately a mutation hook, not a dependency on ``src.tuning``.
        A tuning adapter can call it inside its own transaction boundary.
        """

        self._policy.validate_tuning_parameters(parameters)
        current = self.qnn_config.to_dict()
        candidate = QNNConfig.from_mapping({**current, **dict(parameters)})

        structural_change = (
            candidate.num_quantum_layers != self.num_quantum_layers
            or candidate.entanglement != self.entanglement
        )
        candidate_circuit = self._build_circuit(candidate)
        candidate_encoder = self._build_encoder(candidate)
        candidate_policy = self._build_policy(candidate)
        candidate_optimizer = self._build_measurement_optimizer(candidate)

        if candidate_circuit.weight_shape != self._circuit.weight_shape:
            baseline_rng = copy.deepcopy(self._rng.bit_generator.state)
            try:
                candidate_weights = self._rng.uniform(
                    low=-np.pi,
                    high=np.pi,
                    size=candidate_circuit.weight_shape,
                ).astype(np.float64)
            except Exception:
                self._rng.bit_generator.state = baseline_rng
                raise
            reset_step = True
        else:
            candidate_weights = candidate_circuit.validate_weights(
                self.quantum_weights
            )
            reset_step = False

        self.qnn_config = candidate
        self._circuit = candidate_circuit
        self._encoder = candidate_encoder
        self._policy = candidate_policy
        self.measurement_optimizer = candidate_optimizer
        self.qnn_metrics = candidate_optimizer.metrics
        self.evaluator = self.qnn_metrics
        self.quantum_weights = np.array(candidate_weights, copy=True)
        if reset_step:
            self._training_step = 0
        self._last_gradient = None

        if structural_change:
            self.logger.info(
                "QNN tuning changed circuit structure | layers=%d | entanglement=%s",
                self.num_quantum_layers,
                self.entanglement,
            )
        return candidate.to_dict()

    # ------------------------------------------------------------------
    # Circuit operations
    # ------------------------------------------------------------------
    @staticmethod
    def _define_quantum_gates() -> dict[str, np.ndarray]:
        return {
            "H": np.array(HADAMARD, copy=True),
            "X": np.array(PAULI_X, copy=True),
            "Y": np.array(PAULI_Y, copy=True),
            "Z": np.array(PAULI_Z, copy=True),
        }

    @staticmethod
    def _rx_gate(theta: float) -> np.ndarray:
        return rx(theta)

    @staticmethod
    def _ry_gate(theta: float) -> np.ndarray:
        return ry(theta)

    @staticmethod
    def _rz_gate(theta: float) -> np.ndarray:
        return rz(theta)

    def _controlled_not_gate(
        self,
        control_qubit: int,
        target_qubit: int,
        num_qubits: int | None = None,
    ) -> np.ndarray:
        total = self.num_qubits if num_qubits is None else positive_int(
            num_qubits, "num_qubits"
        )
        for name, value in (
            ("control_qubit", control_qubit),
            ("target_qubit", target_qubit),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise QNNInputError(f"{name} must be an integer")
            if value < 0 or value >= total:
                raise QNNInputError(f"{name} is out of range")
        if control_qubit == target_qubit:
            raise QNNInputError("control_qubit and target_qubit must differ")
        return np.array(CNOT, copy=True)

    def _apply_gate(
        self,
        state: Any,
        gate: Any,
        target_qubits: Sequence[int],
    ) -> np.ndarray:
        return apply_gate(
            state,
            gate,
            target_qubits,
            num_qubits=self.num_qubits,
        )

    def _quantum_layer(self, input_state: Any, layer_weights: Any) -> np.ndarray:
        state = self._encoder.encode_state(input_state, name="input_state")
        return self._circuit.apply_layer(state, layer_weights)

    def forward_sequence(
        self,
        input_sequence: Sequence[Any],
        *,
        weights: Any | None = None,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        inputs = self._normalize_sequence(input_sequence, name="input_sequence")
        parameters = self._circuit.validate_weights(
            self.quantum_weights if weights is None else weights
        )
        self._validate_working_set(len(inputs) + 2, operation="forward_sequence")
        outputs = [self._circuit.forward(state, parameters) for state in inputs]
        return outputs, np.array(outputs[-1], copy=True)

    def _qrnn_forward(
        self,
        input_sequence: Sequence[Any],
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Compatibility alias.

        The repaired model is deliberately non-recurrent; the former averaging
        of an input state and a hidden state was not a unitary quantum update.
        """
        return self.forward_sequence(input_sequence)

    def _initialize_hidden_state(self) -> np.ndarray:
        return zero_state(self.num_qubits)

    # ------------------------------------------------------------------
    # Metrics and optimization
    # ------------------------------------------------------------------
    def _loss_for_weights(
        self,
        weights: np.ndarray,
        inputs: Sequence[np.ndarray],
        targets: Sequence[np.ndarray],
    ) -> float:
        outputs = [self._circuit.forward(state, weights) for state in inputs]
        return float(self.qnn_metrics.evaluate(outputs, targets))

    def _parameter_shift_gradient(
        self,
        loss_fn: Any,
        weights: Any,
        epsilon: float | None = None,
    ) -> np.ndarray:
        parameters = self._circuit.validate_weights(weights)
        shift = (
            self.measurement_optimizer.parameter_shift
            if epsilon is None
            else _finite_scalar(epsilon, "epsilon")
        )
        return parameter_shift_gradient(
            loss_fn,
            parameters,
            loss_name=self.measurement_optimizer.loss,
            shift=shift,
        )

    def _finite_difference_gradient(
        self,
        loss_fn: Any,
        weights: Any,
        epsilon: float | None = None,
    ) -> np.ndarray:
        parameters = self._circuit.validate_weights(weights)
        step = (
            self.measurement_optimizer.finite_difference_step
            if epsilon is None
            else positive_float(epsilon, "epsilon")
        )
        return finite_difference_gradient(loss_fn, parameters, step=step)

    def _compute_gradient(
        self,
        inputs: Sequence[np.ndarray],
        targets: Sequence[np.ndarray],
    ) -> np.ndarray:
        def loss_fn(candidate: np.ndarray) -> float:
            return self._loss_for_weights(candidate, inputs, targets)

        gradient = self.measurement_optimizer.gradient(
            loss_fn,
            self.quantum_weights,
        )
        self._check_gradient_health(gradient)
        return gradient

    @staticmethod
    def _check_gradient_health(gradient: Any) -> None:
        values = np.asarray(gradient, dtype=np.float64)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise QNNInputError("QNN gradient is empty or non-finite")

    def train_task(
        self,
        task: QNNTask | Mapping[str, Any] | Any,
        *,
        steps: int = 1,
    ) -> dict[str, Any]:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise QNNInputError("steps must be a positive integer")

        normalized_task = QNNTask.from_value(task)
        if normalized_task.target_outputs is None:
            raise QNNInputError("training requires target_outputs")

        inputs = self._normalize_sequence(
            normalized_task.input_sequences,
            name="inputs",
        )
        targets = self._normalize_sequence(
            normalized_task.target_outputs,
            name="targets",
        )
        if len(inputs) != len(targets):
            raise QNNInputError(
                "training inputs and targets must contain the same number of states"
            )

        self._validate_working_set(
            4 * len(inputs),
            operation="training",
        )
        self._policy.validate_training_work(
            sequence_length=len(inputs),
            steps=steps,
        )

        initial_loss = self._loss_for_weights(
            self.quantum_weights,
            inputs,
            targets,
        )
        baseline_weights = np.array(self.quantum_weights, copy=True)
        baseline_step = self._training_step
        baseline_gradient = (
            None
            if self._last_gradient is None
            else np.array(self._last_gradient, copy=True)
        )

        try:
            for _ in range(steps):
                gradient = self._compute_gradient(inputs, targets)
                with np.errstate(over="raise", invalid="raise"):
                    updated = self.quantum_weights - self.learning_rate * gradient
                self.quantum_weights = np.array(
                    self._circuit.validate_weights(updated),
                    copy=True,
                )
                self._last_gradient = np.array(gradient, copy=True)
                self._training_step += 1

            outputs = [
                self._circuit.forward(state, self.quantum_weights)
                for state in inputs
            ]
            metrics = self.qnn_metrics.evaluate_sequence(outputs, targets)
            gradient = self._last_gradient
            if gradient is None:
                raise QNNInputError("training completed without a gradient")
            metrics.update(
                {
                    "initial_loss": float(initial_loss),
                    "gradient_norm": float(np.linalg.norm(gradient)),
                    "gradient_variance": float(np.var(gradient)),
                }
            )
            return {
                "outputs": outputs,
                "probabilities": [
                    self.measurement_optimizer.probabilities(state)
                    for state in outputs
                ],
                "metrics": metrics,
                "loss": metrics["loss"],
                "training_step": self._training_step,
            }
        except Exception:
            self.quantum_weights = baseline_weights
            self._training_step = baseline_step
            self._last_gradient = baseline_gradient
            raise

    def _evaluate_performance(
        self,
        output_sequence: Sequence[Any],
        task: Any,
    ) -> list[float]:
        normalized = QNNTask.from_value(task)
        if normalized.target_outputs is None:
            raise QNNInputError("evaluation requires target_outputs")
        targets = self._normalize_sequence(
            normalized.target_outputs,
            name="targets",
        )
        return [self.qnn_metrics.evaluate(output_sequence, targets)]

    # ------------------------------------------------------------------
    # BaseAgent task surface
    # ------------------------------------------------------------------
    def _task_values(
        self,
        task_data: Any,
    ) -> tuple[str, tuple[QNNTask, ...], int]:
        if isinstance(task_data, Mapping):
            mode = str(
                task_data.get("mode", task_data.get("operation", "infer"))
            ).strip().casefold()
            steps = task_data.get("steps", 1)
            if "tasks" in task_data:
                raw_tasks = task_data["tasks"]
                if not isinstance(raw_tasks, Sequence) or isinstance(
                    raw_tasks, (str, bytes, bytearray)
                ):
                    raise QNNInputError("tasks must be a sequence")
                tasks = tuple(QNNTask.from_value(item) for item in raw_tasks)
            else:
                tasks = (QNNTask.from_value(task_data),)
        else:
            mode = "infer"
            steps = 1
            tasks = (QNNTask.from_value(task_data),)

        if mode in {"inference", "predict"}:
            mode = "infer"
        if mode not in {"infer", "evaluate", "train"}:
            raise QNNInputError("mode must be infer, evaluate, or train")
        if not tasks:
            raise QNNInputError("at least one QNN task is required")

        self._policy.validate_tasks(tasks)
        for index, task in enumerate(tasks):
            self._policy.validate_sequence_length(
                len(task.input_sequences),
                name=f"tasks[{index}].input_sequences",
            )

        if isinstance(steps, bool) or not isinstance(steps, int):
            raise QNNInputError("steps must be an integer")
        if mode == "train" and steps <= 0:
            raise QNNInputError("training steps must be positive")

        total_states = sum(len(task.input_sequences) for task in tasks)
        self._validate_working_set(
            max(1, 4 * total_states),
            operation=f"{mode} request",
        )
        return mode, tasks, steps

    def _execute_qnn_task(
        self,
        task: QNNTask,
        mode: str,
        steps: int,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        if mode == "train":
            result = self.train_task(task, steps=steps)
        else:
            inputs = self._normalize_sequence(task.input_sequences, name="inputs")
            outputs = [
                self._circuit.forward(state, self.quantum_weights)
                for state in inputs
            ]
            probabilities = [
                self.measurement_optimizer.probabilities(state)
                for state in outputs
            ]
            metrics: dict[str, float] = {
                "norm_error": float(
                    max(
                        abs(float(np.linalg.norm(state)) - 1.0)
                        for state in outputs
                    )
                )
            }

            if mode == "evaluate":
                if task.target_outputs is None:
                    raise QNNInputError("evaluation requires target_outputs")
                targets = self._normalize_sequence(
                    task.target_outputs,
                    name="targets",
                )
                if len(outputs) != len(targets):
                    raise QNNInputError(
                        "evaluation outputs and targets must be aligned"
                    )
                metrics.update(
                    self.qnn_metrics.evaluate_sequence(outputs, targets)
                )

            result = {
                "outputs": outputs,
                "probabilities": probabilities,
                "metrics": metrics,
            }
            if "loss" in metrics:
                result["loss"] = metrics["loss"]

        result["latency_seconds"] = float(time.perf_counter() - started)
        result["metadata"] = dict(task.metadata)
        return result

    def perform_task(self, task_data: Any) -> dict[str, Any]:
        mode, tasks, steps = self._task_values(task_data)

        baseline_model = self.state_dict() if mode == "train" else None
        baseline_gradient = (
            None
            if self._last_gradient is None
            else np.array(self._last_gradient, copy=True)
        )

        try:
            results = [
                self._execute_qnn_task(task, mode, steps)
                for task in tasks
            ]
        except Exception:
            if baseline_model is not None:
                self.load_state_dict(baseline_model)
                self._last_gradient = baseline_gradient
            raise

        metric_names = (
            set.intersection(
                *(set(result["metrics"]) for result in results)
            )
            if results
            else set()
        )
        aggregate_metrics = {
            name: float(
                np.mean([result["metrics"][name] for result in results])
            )
            for name in sorted(metric_names)
        }

        response: dict[str, Any] = {
            "status": "success",
            "agent": self.name,
            "model_kind": "parameterized_statevector_circuit",
            "mode": mode,
            "task_count": len(results),
            "results": results,
            "metrics": aggregate_metrics,
            "training_step": self._training_step,
            "statevector_bytes": self.estimated_statevector_bytes,
            "parameter_bytes": self.parameter_bytes,
            "parameter_count": self.qnn_config.parameter_count,
        }
        if "loss" in aggregate_metrics:
            response["loss"] = aggregate_metrics["loss"]
        if len(results) == 1:
            for key in ("outputs", "probabilities", "latency_seconds"):
                response[key] = results[0][key]

        # Feed BaseAgent's lightweight metric surface without making telemetry
        # success a correctness requirement for QNN execution.
        for name, value in aggregate_metrics.items():
            try:
                self.performance_metrics[name].append(value)
            except Exception as exc:
                self._mark_runtime_degraded(
                    "telemetry",
                    "qnn.performance_metrics",
                    exc,
                )

        self._publish_qnn_summary(response)
        return response

    def predict(self, input_data: Any, context: Any = None) -> dict[str, Any]:
        if isinstance(input_data, Mapping):
            payload = dict(input_data)
            payload["mode"] = "infer"
        else:
            payload = {
                "mode": "infer",
                "input_sequences": input_data,
            }

        if context is not None:
            metadata = dict(payload.get("metadata") or {})
            metadata["context"] = context
            payload["metadata"] = metadata
        return self.perform_task(payload)

    def run_task(self, task: Any) -> list[np.ndarray]:
        normalized = QNNTask.from_value(task)
        outputs, _ = self.forward_sequence(normalized.input_sequences)
        return outputs

    def train(self, tasks: Any, *, steps: int = 1) -> dict[str, Any]:
        if isinstance(tasks, (QNNTask, Mapping)) or hasattr(
            tasks, "input_sequences"
        ):
            payload_tasks = [tasks]
        else:
            if isinstance(tasks, (str, bytes, bytearray)) or not isinstance(
                tasks, Sequence
            ):
                raise QNNInputError("tasks must be a QNN task or sequence of tasks")
            payload_tasks = list(tasks)
        return self.perform_task(
            {
                "mode": "train",
                "tasks": payload_tasks,
                "steps": steps,
            }
        )

    def _born_sample(
        self,
        state: Any,
        num_samples: int = 1,
    ) -> np.ndarray:
        sample_count = positive_int(num_samples, "num_samples")
        encoded_state = self._encoder.encode_state(
            state,
            name="sample_state",
        )
        return self.measurement_optimizer.sample(
            encoded_state,
            rng=self._rng,
            num_samples=sample_count,
        )

    # ------------------------------------------------------------------
    # Effective numerical profile
    # ------------------------------------------------------------------
    @staticmethod
    def _numerical_profile_for(
        encoder: StateVectorEncoder,
        measurement_optimizer: QuantumMeasurementOptimizer,
        policy: QuantumExecutionPolicy,
    ) -> dict[str, Any]:
        policy_limits = {
            "max_statevector_bytes": positive_int(
                policy.policy_config["max_statevector_bytes"],
                "quantum_policy.max_statevector_bytes",
            ),
            "max_parameter_bytes": positive_int(
                policy.policy_config["max_parameter_bytes"],
                "quantum_policy.max_parameter_bytes",
            ),
            **{
                name: getattr(policy, name)
                for name in (
                    "max_working_set_bytes",
                    "max_sequence_length",
                    "max_tasks_per_request",
                    "max_gradient_evaluations",
                    "max_training_steps",
                )
            },
        }
        return {
            "encoding": {
                "norm_tolerance": encoder.tolerance,
                "normalize_inputs": encoder.normalize,
            },
            "measurement_optimization": {
                "loss": measurement_optimizer.loss,
                "gradient_method": measurement_optimizer.gradient_method,
                "probability_tolerance": (
                    measurement_optimizer.probability_tolerance
                ),
                "parameter_shift": measurement_optimizer.parameter_shift,
                "finite_difference_step": (
                    measurement_optimizer.finite_difference_step
                ),
                "gradient_clip_norm": (
                    measurement_optimizer.gradient_clip_norm
                ),
            },
            "policy": {
                **policy_limits,
                "allowed_tuning_parameters": sorted(
                    policy.allowed_tuning_parameters
                ),
            },
        }

    def _numerical_profile(self) -> dict[str, Any]:
        return self._numerical_profile_for(
            self._encoder,
            self.measurement_optimizer,
            self._policy,
        )

    @classmethod
    def _fingerprint_components(
        cls,
        config: QNNConfig,
        encoder: StateVectorEncoder,
        measurement_optimizer: QuantumMeasurementOptimizer,
        policy: QuantumExecutionPolicy,
    ) -> str:
        effective_config = {
            "agent": config.to_dict(),
            "numerical_profile": cls._numerical_profile_for(
                encoder,
                measurement_optimizer,
                policy,
            ),
        }
        canonical = json.dumps(
            json_safe(effective_config),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _configuration_fingerprint(self) -> str:
        return self._fingerprint_components(
            self.qnn_config,
            self._encoder,
            self.measurement_optimizer,
            self._policy,
        )

    # ------------------------------------------------------------------
    # Shared-memory observability
    # ------------------------------------------------------------------
    def _publish_qnn_summary(self, response: Mapping[str, Any]) -> None:
        if not self._shared_memory_config["enabled"]:
            return

        payload = {
            "agent_id": self.agent_id,
            "model_kind": response.get("model_kind"),
            "mode": response.get("mode"),
            "status": response.get("status"),
            "task_count": response.get("task_count"),
            "training_step": self._training_step,
            "metrics": dict(response.get("metrics") or {}),
            "statevector_bytes": self.estimated_statevector_bytes,
            "parameter_bytes": self.parameter_bytes,
            "parameter_count": self.qnn_config.parameter_count,
            "config_fingerprint": self._configuration_fingerprint(),
            "updated_at": time.time(),
        }
        safe_payload = json_safe(payload)
        summary_key = (
            f"{self._shared_memory_config['summary_key_prefix']}:"
            f"{self.agent_id}"
        )
        ttl = self._shared_memory_config["ttl_seconds"]

        self._run_optional_runtime_operation(
            "telemetry",
            "shared_memory.qnn_summary",
            lambda: self.shared_memory.set(
                summary_key,
                safe_payload,
                ttl=ttl,
            ),
        )

        if self._shared_memory_config["publish_notifications"]:
            event_payload = {
                "event": "qnn_task_completed",
                **safe_payload,
            }
            self._run_optional_runtime_operation(
                "telemetry",
                "shared_memory.qnn_event",
                lambda: self.shared_memory.publish(
                    self._shared_memory_config["event_channel"],
                    event_payload,
                ),
            )

    # ------------------------------------------------------------------
    # Model / agent state hooks for external checkpointing and tuning
    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "quantum_weights": np.array(
                self.quantum_weights,
                copy=True,
            ),
            "training_step": np.array(
                [self._training_step],
                dtype=np.int64,
            ),
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        strict: bool = True,
    ) -> None:
        if not isinstance(state, Mapping):
            raise QNNInputError("QNN model state must be a mapping")
        if not isinstance(strict, bool):
            raise TypeError("strict must be a boolean")

        keys = set(state)
        missing = self._MODEL_STATE_FIELDS - keys
        unknown = keys - self._MODEL_STATE_FIELDS
        if missing or (strict and unknown):
            raise QNNInputError(
                "invalid QNN model state; "
                f"missing={sorted(str(item) for item in missing)}, "
                f"unknown={sorted(str(item) for item in unknown)}"
            )

        candidate_weights = np.array(
            self._circuit.validate_weights(state["quantum_weights"]),
            copy=True,
        )
        raw_step = np.asarray(state["training_step"])
        if raw_step.size != 1:
            raise QNNInputError("training_step must contain one integer")
        step_item = raw_step.reshape(-1)[0]
        step_array = np.asarray(step_item)
        if np.issubdtype(step_array.dtype, np.bool_) or not np.issubdtype(
            step_array.dtype,
            np.integer,
        ):
            raise QNNInputError("training_step must be an integer")
        candidate_step = int(step_item)
        if candidate_step < 0:
            raise QNNInputError("training_step must be non-negative")

        self.quantum_weights = candidate_weights
        self._training_step = candidate_step
        self._last_gradient = None

    def agent_state(self) -> dict[str, Any]:
        return {
            "schema": _CHECKPOINT_SCHEMA,
            "configuration": self.qnn_config.to_dict(),
            "numerical_profile_fingerprint": self._configuration_fingerprint(),
            "training_step": self._training_step,
            "rng_bit_generator": self._rng.bit_generator.__class__.__name__,
            "rng_state": json_safe(
                copy.deepcopy(self._rng.bit_generator.state)
            ),
        }

    def load_agent_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise QNNInputError("QNN agent state must be a mapping")
        if state.get("schema") != _CHECKPOINT_SCHEMA:
            raise QNNInputError("unsupported QNN agent-state schema")

        checkpoint_config = QNNConfig.from_mapping(
            state.get("configuration")
        )
        if (
            checkpoint_config.num_qubits != self.num_qubits
            or checkpoint_config.num_quantum_layers != self.num_quantum_layers
        ):
            raise QNNInputError(
                "checkpoint QNN architecture does not match the active agent"
            )

        state_training_step = state.get("training_step")
        if (
            isinstance(state_training_step, bool)
            or not isinstance(state_training_step, (int, np.integer))
            or int(state_training_step) < 0
        ):
            raise QNNInputError(
                "checkpoint agent-state training_step must be non-negative"
            )
        if int(state_training_step) != self._training_step:
            raise QNNInputError(
                "checkpoint model and agent-state training_step values disagree"
            )

        generator_name = state.get("rng_bit_generator")
        if generator_name != self._rng.bit_generator.__class__.__name__:
            raise QNNInputError(
                "checkpoint RNG bit generator is incompatible"
            )
        rng_state = state.get("rng_state")
        if not isinstance(rng_state, Mapping):
            raise QNNInputError(
                "checkpoint rng_state must be a mapping"
            )

        candidate_rng = np.random.Generator(
            self._rng.bit_generator.__class__()
        )
        try:
            candidate_rng.bit_generator.state = json_safe(rng_state)
        except (TypeError, ValueError) as exc:
            raise QNNInputError(
                "checkpoint rng_state is invalid"
            ) from exc

        candidate_circuit = self._build_circuit(checkpoint_config)
        candidate_circuit.validate_weights(self.quantum_weights)
        candidate_encoder = self._build_encoder(checkpoint_config)
        candidate_policy = self._build_policy(checkpoint_config)
        candidate_optimizer = self._build_measurement_optimizer(
            checkpoint_config
        )

        profile_fingerprint = state.get(
            "numerical_profile_fingerprint"
        )
        if profile_fingerprint is not None:
            if (
                not isinstance(profile_fingerprint, str)
                or not profile_fingerprint
            ):
                raise QNNInputError(
                    "checkpoint numerical_profile_fingerprint must be "
                    "a non-empty string"
                )
            candidate_profile = self._fingerprint_components(
                checkpoint_config,
                candidate_encoder,
                candidate_optimizer,
                candidate_policy,
            )
            if profile_fingerprint != candidate_profile:
                raise QNNInputError(
                    "checkpoint numerical configuration is incompatible "
                    "with the active QNN subsystem configuration"
                )

        self.qnn_config = checkpoint_config
        self._circuit = candidate_circuit
        self._encoder = candidate_encoder
        self._policy = candidate_policy
        self.measurement_optimizer = candidate_optimizer
        self.qnn_metrics = candidate_optimizer.metrics
        self.evaluator = self.qnn_metrics
        self._rng = candidate_rng

    def checkpoint_components(self) -> dict[str, Any]:
        return {
            "model": self.state_dict(),
            "agent_state": self.agent_state(),
        }

    def save_checkpoint(
        self,
        manager: Any,
        *,
        version: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        overwrite: bool | None = None,
    ) -> Any:
        """Save through an injected CheckpointManager-like boundary.

        QNNAgent intentionally does not import ``src.checkpointing``.
        """
        save_components = getattr(manager, "save_components", None)
        if not callable(save_components):
            raise TypeError(
                "checkpoint manager must expose save_components()"
            )

        try:
            result = save_components(
                self.checkpoint_components(),
                version=version,
                codec_ids={
                    "model": "numpy",
                    "agent_state": "agent-state",
                },
                step=self._training_step,
                metrics=self._latest_checkpoint_metrics(),
                metadata={
                    "agent_type": self.name,
                    "state_schema": _CHECKPOINT_SCHEMA,
                    **dict(metadata or {}),
                },
                overwrite=overwrite,
            )
        except Exception as exc:
            self._mark_runtime_degraded(
                "persistence",
                "qnn.checkpoint.save",
                exc,
            )
            raise

        self._mark_runtime_recovered(
            "persistence",
            "qnn.checkpoint.save",
        )
        return result

    def load_checkpoint(
        self,
        manager: Any,
        *,
        version: str | None = None,
    ) -> Any:
        """Load through an injected CheckpointManager-like boundary atomically."""
        load_components = getattr(manager, "load_components", None)
        if not callable(load_components):
            raise TypeError(
                "checkpoint manager must expose load_components()"
            )

        baseline_model = self.state_dict()
        baseline_agent_state = self.agent_state()
        try:
            result = load_components(
                version,
                components=("model", "agent_state"),
                expected_codecs={
                    "model": "numpy",
                    "agent_state": "agent-state",
                },
            )
            components = getattr(result, "components", None)
            if not isinstance(components, Mapping):
                raise QNNInputError(
                    "checkpoint load result does not contain components"
                )
            if "model" not in components or "agent_state" not in components:
                raise QNNInputError(
                    "checkpoint load result is missing required QNN components"
                )

            self.load_state_dict(components["model"])
            self.load_agent_state(components["agent_state"])
        except Exception as exc:
            # Restore both model and agent state. Baseline agent_state was
            # generated from the pre-load model, so load the model first.
            self.load_state_dict(baseline_model)
            self.load_agent_state(baseline_agent_state)
            self._mark_runtime_degraded(
                "persistence",
                "qnn.checkpoint.load",
                exc,
            )
            raise

        self._mark_runtime_recovered(
            "persistence",
            "qnn.checkpoint.load",
        )
        return result

    def _latest_checkpoint_metrics(self) -> dict[str, float]:
        try:
            loss_values = tuple(self.performance_metrics.get("loss", ()))
        except Exception:
            return {}
        return (
            {"loss": float(loss_values[-1])}
            if loss_values
            else {}
        )

    # ------------------------------------------------------------------
    # Optional visualization; matplotlib remains lazy
    # ------------------------------------------------------------------
    def visualize_output(
        self,
        output_sequence: Sequence[Any],
        *,
        show: bool = False,
    ) -> Any:
        if self.num_qubits == 1:
            return self.visualize_bloch(
                output_sequence,
                show=show,
            )
        return self.visualize_amplitudes(
            output_sequence,
            show=show,
        )

    def visualize_bloch(
        self,
        output_sequence: Sequence[Any],
        *,
        show: bool = False,
    ) -> Any:
        import matplotlib.pyplot as plt

        states = self._normalize_sequence(
            output_sequence,
            name="output_sequence",
        )
        if self.num_qubits != 1:
            raise QNNInputError(
                "Bloch visualization requires exactly one qubit"
            )

        figure = plt.figure(figsize=(8, 6))
        axes = figure.add_subplot(111, projection="3d")
        for state in states:
            x, y, z = self._bloch_coordinates(state)
            axes.scatter(x, y, z, s=50) # type: ignore
        axes.set(
            title="Bloch Sphere Projection",
            xlabel="X",
            ylabel="Y",
            zlabel="Z",
        )
        if show:
            plt.show()
        return figure

    @staticmethod
    def _bloch_coordinates(
        state: Any,
    ) -> tuple[float, float, float]:
        amplitudes = np.asarray(
            state,
            dtype=np.complex128,
        )
        if amplitudes.shape != (2,):
            raise QNNInputError(
                "Bloch coordinates require a one-qubit state"
            )
        a, b = amplitudes
        coherence = np.conjugate(a) * b
        return (
            float(2.0 * coherence.real),
            float(2.0 * coherence.imag),
            float(abs(a) ** 2 - abs(b) ** 2),
        )

    def visualize_amplitudes(
        self,
        output_sequence: Sequence[Any],
        *,
        show: bool = False,
    ) -> list[Any]:
        import matplotlib.pyplot as plt

        states = self._normalize_sequence(
            output_sequence,
            name="output_sequence",
        )
        figures = []
        for index, state in enumerate(states):
            figure, axes = plt.subplots(figsize=(7, 3))
            axes.bar(range(state.size), np.abs(state))
            axes.set(
                title=f"QNN Output | Step {index}",
                xlabel="Basis State Index",
                ylabel="Amplitude magnitude",
            )
            figure.tight_layout()
            figures.append(figure)
        if show:
            plt.show()
        return figures


Task = QNNTask

__all__ = [
    "PerformanceEvaluator",
    "QNNAgent",
    "QNNConfig",
    "QNNTask",
    "QuantumGate",
    "StateVectorCircuit",
    "Task",
]


if __name__ == "__main__":
    basis_zero = np.array(
        [1.0, 0.0, 0.0, 0.0],
        dtype=np.complex128,
    )
    agent = QNNAgent(
        config={
            "num_qubits": 2,
            "num_quantum_layers": 1,
            "seed": 7,
        }
    )
    result = agent.perform_task(
        {
            "mode": "infer",
            "input_sequences": [basis_zero],
        }
    )
    print(
        {
            "status": result["status"],
            "model_kind": result["model_kind"],
            "norm_error": result["metrics"]["norm_error"],
        }
    )
