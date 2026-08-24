"""SLAI QNN agent backed by an explicit NumPy state-vector circuit.

The agent owns state validation, parameterized unitary evolution, Born
measurement, QNN-specific optimization, and model state. Agent selection,
cross-agent comparison, tuning search, and durable checkpoint storage remain
owned by their existing SLAI subsystems and are reached through opt-in
adapters.

This is a classical state-vector simulation of a parameterized quantum
circuit. It does not claim quantum speed-up, hardware execution, or a quantum
recurrent architecture.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
import numpy as np


from collections.abc import Mapping, Sequence
from typing import Any

from .base.utils.main_config_loader import get_config_section, load_global_config
from .base_agent import BaseAgent
from .qnn.quantum_encoding import *
from .qnn.quantum_memory import *
from .qnn.quantum_mno import *
from .qnn.quantum_policy import *
from .qnn.utils.quantum_helpers import *
from .qnn.utils.quantum_errors import *

_CHECKPOINT_SCHEMA = "slai.qnn-agent.state.v1"


class QNNAgent(BaseAgent):
    """Parameterized state-vector QNN with explicit SLAI runtime boundaries."""

    capabilities = (
        "quantum_state_simulation",
        "quantum_circuit_inference",
        "quantum_circuit_training",
        "quantum_state_evaluation",
    )

    _MODEL_STATE_FIELDS = frozenset({"quantum_weights", "training_step"})
    _DEFAULT_SHARED_MEMORY_CONFIG = {
        "enabled": True,
        "ttl_seconds": None,
        "publish_notifications": False,
        "summary_key_prefix": "qnn:summary",
        "event_channel": "qnn.events",
    }

    def __init__(
        self,
        shared_memory: Any = None,
        agent_factory: Any = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            config=config,
        )

        # BaseAgent already owns ``shared_memory``, ``agent_factory``, and the
        # loaded agents_config.yaml snapshot. Do not replace those resolved
        # dependencies with the raw constructor arguments.
        resolved = dict(
            get_config_section(
                "qnn_agent",
                config=self.config,
            )
            or {}
        )
        if config:
            if not isinstance(config, Mapping):
                raise QNNConfigurationError("QNN config override must be a mapping")
            resolved.update(dict(config))

        self.qnn_config = from_mapping(resolved)
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
        # Compatibility alias for callers of the former monolithic QNN.
        self.evaluator = self.qnn_metrics
        self.quantum_weights = self._initialize_quantum_weights()
        self._training_step = 0
        self._last_gradient: np.ndarray | None = None
        self.gate_definitions = self._define_quantum_gates()

    # ------------------------------------------------------------------
    # Configuration and state contracts
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
        return StateVectorEncoder(
            state_dimension=config.state_dimension,
        )

    @staticmethod
    def _build_policy(config: QNNConfig) -> QuantumExecutionPolicy:
        """Delegate QNN resource-policy construction to its owning module."""

        return QuantumExecutionPolicy.from_config(config)

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
    def _resolve_shared_memory_config(
        cls,
        value: Any,
    ) -> dict[str, Any]:
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
            if isinstance(ttl, bool) or not isinstance(ttl, (int, float, np.number)):
                raise QNNConfigurationError(
                    "qnn_agent.shared_memory.ttl_seconds must be numeric or null"
                )
            ttl = float(ttl)
            if not np.isfinite(ttl) or ttl <= 0.0:
                raise QNNConfigurationError(
                    "qnn_agent.shared_memory.ttl_seconds must be finite and positive"
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
        """Bytes occupied by one complex128 state vector, excluding scratch space."""

        return self.qnn_config.statevector_bytes

    @property
    def parameter_bytes(self) -> int:
        """Bytes occupied by the float64 rotation-parameter tensor."""

        return self.qnn_config.parameter_bytes

    @property
    def training_step(self) -> int:
        return self._training_step

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
        values = tuple(sequence)
        encoded = self._encoder.encode_sequence(values, name=name)
        self._policy.validate_sequence_length(len(encoded), name=name)
        return encoded

    def _validate_working_set(self, state_slots: int, *, operation: str) -> None:
        """Reject work whose conservative array budget exceeds policy."""

        self._policy.validate_working_set(state_slots, operation=operation)

    def apply_tuning_parameters(self, parameters: Mapping[str, Any]) -> dict[str, Any]:
        """Apply only declared QNN tunables to an isolated tuning candidate."""

        self._policy.validate_tuning_parameters(parameters)
        current = self.qnn_config.to_dict()
        candidate = QNNConfig.from_mapping({**current, **dict(parameters)})
        structural_change = candidate.num_quantum_layers != self.num_quantum_layers
        candidate_circuit = self._build_circuit(candidate)
        candidate_encoder = self._build_encoder(candidate)
        candidate_policy = self._build_policy(candidate)
        candidate_measurement_optimizer = self._build_measurement_optimizer(candidate)
        if structural_change:
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
        else:
            candidate_weights = candidate_circuit.validate_weights(self.quantum_weights)
        self.qnn_config = candidate
        self._circuit = candidate_circuit
        self._encoder = candidate_encoder
        self._policy = candidate_policy
        self.measurement_optimizer = candidate_measurement_optimizer
        self.qnn_metrics = candidate_measurement_optimizer.metrics
        self.evaluator = self.qnn_metrics
        self.quantum_weights = np.array(candidate_weights, copy=True)
        if structural_change:
            self._training_step = 0
        self._last_gradient = None
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
        """Return the local CNOT; qubit order is supplied to ``_apply_gate``."""

        total = self.num_qubits if num_qubits is None else num_qubits
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
        """Apply the same parameterized unitary circuit to each encoded state."""

        inputs = self._normalize_sequence(input_sequence, name="input_sequence")
        parameters = self._circuit.validate_weights(
            self.quantum_weights if weights is None else weights
        )
        outputs = [self._circuit.forward(state, parameters) for state in inputs]
        return outputs, np.array(outputs[-1], copy=True)

    def _qrnn_forward(
        self,
        input_sequence: Sequence[Any],
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Compatibility alias; the repaired model is deliberately non-recurrent."""

        return self.forward_sequence(input_sequence)

    def _initialize_hidden_state(self) -> np.ndarray:
        """Compatibility helper returning a valid ``|0...0>`` state."""

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
        """Differentiate a compatible expectation-derived loss by parameter shift."""

        parameters = self._circuit.validate_weights(weights)
        shift = (
            self.measurement_optimizer.parameter_shift
            if epsilon is None
            else float(epsilon)
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
            else float(epsilon)
        )
        return finite_difference_gradient(loss_fn, parameters, step=step)

    def _compute_gradient(
        self,
        inputs: Sequence[np.ndarray],
        targets: Sequence[np.ndarray],
    ) -> np.ndarray:
        def loss_fn(candidate: np.ndarray) -> float:
            return self._loss_for_weights(candidate, inputs, targets)

        return self.measurement_optimizer.gradient(loss_fn, self.quantum_weights)

    @staticmethod
    def _check_gradient_health(gradient: Any) -> None:
        values = np.asarray(gradient, dtype=np.float64)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise FloatingPointError("QNN gradient is empty or non-finite")

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
            normalized_task.input_sequences, name="inputs"
        )
        targets = self._normalize_sequence(
            normalized_task.target_outputs, name="targets"
        )
        self._validate_working_set(4 * len(inputs), operation="training")
        self._policy.validate_training_work(
            sequence_length=len(inputs),
            steps=steps,
        )

        initial_loss = self._loss_for_weights(self.quantum_weights, inputs, targets)
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
                self._circuit.forward(state, self.quantum_weights) for state in inputs
            ]
            metrics = self.qnn_metrics.evaluate_sequence(outputs, targets)
            gradient = self._last_gradient
            assert gradient is not None
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
        targets = self._normalize_sequence(normalized.target_outputs, name="targets")
        return [self.qnn_metrics.evaluate(output_sequence, targets)]

    # ------------------------------------------------------------------
    # BaseAgent task surface
    # ------------------------------------------------------------------
    def _task_values(self, task_data: Any) -> tuple[str, tuple[QNNTask, ...], int]:
        if isinstance(task_data, Mapping):
            mode = (
                str(task_data.get("mode", task_data.get("operation", "infer")))
                .strip()
                .casefold()
            )
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
            4 * total_states,
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
                self._circuit.forward(state, self.quantum_weights) for state in inputs
            ]
            probabilities = [
                self.measurement_optimizer.probabilities(state)
                for state in outputs
            ]
            metrics: dict[str, float] = {
                "norm_error": float(
                    max(abs(float(np.linalg.norm(state)) - 1.0) for state in outputs)
                )
            }
            if mode == "evaluate":
                if task.target_outputs is None:
                    raise QNNInputError("evaluation requires target_outputs")
                targets = self._normalize_sequence(task.target_outputs, name="targets")
                metrics.update(self.qnn_metrics.evaluate_sequence(outputs, targets))
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
            results = [self._execute_qnn_task(task, mode, steps) for task in tasks]
        except Exception:
            if baseline_model is not None:
                self.load_state_dict(baseline_model)
                self._last_gradient = baseline_gradient
            raise
        metric_names = (
            set.intersection(*(set(result["metrics"]) for result in results))
            if results
            else set()
        )
        aggregate_metrics = {
            name: float(np.mean([result["metrics"][name] for result in results]))
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
        self._publish_qnn_summary(response)
        return response

    def predict(self, input_data: Any, context: Any = None) -> dict[str, Any]:
        if isinstance(input_data, Mapping):
            payload = dict(input_data)
            payload["mode"] = "infer"
        else:
            payload = {"mode": "infer", "input_sequences": input_data}
        if context is not None:
            metadata = dict(payload.get("metadata") or {})
            metadata["context"] = context
            payload["metadata"] = metadata
        return self.perform_task(payload)

    def run_task(self, task: Any) -> list[np.ndarray]:
        """Compatibility inference helper returning only output states."""

        normalized = QNNTask.from_value(task)
        outputs, _ = self.forward_sequence(normalized.input_sequences)
        return outputs

    def train(self, tasks: Any, *, steps: int = 1) -> dict[str, Any]:
        if isinstance(tasks, (QNNTask, Mapping)) or hasattr(tasks, "input_sequences"):
            payload_tasks = [tasks]
        else:
            payload_tasks = list(tasks)
        return self.perform_task(
            {"mode": "train", "tasks": payload_tasks, "steps": steps}
        )

    def _born_sample(self, state: Any, num_samples: int = 1) -> np.ndarray:
        if (
            isinstance(num_samples, bool)
            or not isinstance(num_samples, int)
            or num_samples <= 0
        ):
            raise QNNInputError("num_samples must be a positive integer")
        encoded_state = self._encoder.encode_state(state, name="sample_state")
        return self.measurement_optimizer.sample(
            encoded_state,
            rng=self._rng,
            num_samples=num_samples,
        )

    @staticmethod
    def _numerical_profile_for(
        encoder: StateVectorEncoder,
        measurement_optimizer: QuantumMeasurementOptimizer,
        policy: QuantumExecutionPolicy,
    ) -> dict[str, Any]:
        policy_limits = {
            name: policy.policy_config[name]
            for name in (
                "max_statevector_bytes",
                "max_parameter_bytes",
                "max_working_set_bytes",
                "max_sequence_length",
                "max_tasks_per_request",
                "max_gradient_evaluations",
                "max_training_steps",
            )
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
        """Return behavior-defining numerical settings without mutable arrays."""

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
    # Durable state and SLAI checkpoint adapter hooks
    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, np.ndarray]:
        """Return a defensive local state snapshot for external adapters."""

        return {
            "quantum_weights": np.array(self.quantum_weights, copy=True),
            "training_step": np.array([self._training_step], dtype=np.int64),
        }

    def load_state_dict(self, state: Mapping[str, Any], *, strict: bool = True) -> None:
        """Validate and atomically apply a decoded model-state component."""

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
            "rng_state": json_safe(copy.deepcopy(self._rng.bit_generator.state)),
        }

    def load_agent_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("QNN agent state must be a mapping")
        if state.get("schema") != _CHECKPOINT_SCHEMA:
            raise QNNInputError("unsupported QNN agent-state schema")
        checkpoint_config = QNNConfig.from_mapping(state.get("configuration"))
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
            raise QNNInputError("checkpoint RNG bit generator is incompatible")
        rng_state = state.get("rng_state")
        if not isinstance(rng_state, Mapping):
            raise QNNInputError("checkpoint rng_state must be a mapping")
        # AgentStateCodec returns immutable mapping proxies. Convert the
        # complete nested structure back to JSON-native containers before
        # handing it to NumPy's bit generator.
        candidate_rng = np.random.Generator(self._rng.bit_generator.__class__())
        try:
            candidate_rng.bit_generator.state = json_safe(rng_state)
        except (TypeError, ValueError) as exc:
            raise QNNInputError("checkpoint rng_state is invalid") from exc
        candidate_circuit = self._build_circuit(checkpoint_config)
        candidate_circuit.validate_weights(self.quantum_weights)
        candidate_encoder = self._build_encoder(checkpoint_config)
        candidate_policy = self._build_policy(checkpoint_config)
        candidate_measurement_optimizer = self._build_measurement_optimizer(
            checkpoint_config
        )

        profile_fingerprint = state.get("numerical_profile_fingerprint")
        if profile_fingerprint is not None:
            if not isinstance(profile_fingerprint, str) or not profile_fingerprint:
                raise QNNInputError(
                    "checkpoint numerical_profile_fingerprint must be a "
                    "non-empty string"
                )
            candidate_profile = self._fingerprint_components(
                checkpoint_config,
                candidate_encoder,
                candidate_measurement_optimizer,
                candidate_policy,
            )
            if profile_fingerprint != candidate_profile:
                raise QNNInputError(
                    "checkpoint numerical configuration is incompatible with "
                    "the active QNN subsystem configuration"
                )

        self.qnn_config = checkpoint_config
        self._circuit = candidate_circuit
        self._encoder = candidate_encoder
        self._policy = candidate_policy
        self.measurement_optimizer = candidate_measurement_optimizer
        self.qnn_metrics = candidate_measurement_optimizer.metrics
        self.evaluator = self.qnn_metrics
        self._rng = candidate_rng

    def checkpoint_components(self) -> dict[str, Any]:
        return {"model": self.state_dict(), "agent_state": self.agent_state()}

    def save_checkpoint(
        self,
        manager: Any,
        *,
        version: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        overwrite: bool | None = None,
    ) -> Any:
        save_components = getattr(manager, "save_components", None)
        if not callable(save_components):
            raise TypeError("checkpoint manager must expose save_components()")
        try:
            result = save_components(
                self.checkpoint_components(),
                version=version,
                codec_ids={"model": "numpy", "agent_state": "agent-state"},
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
            self._mark_runtime_degraded("persistence", "qnn.checkpoint.save", exc)
            raise
        self._mark_runtime_recovered("persistence", "qnn.checkpoint.save")
        return result

    def load_checkpoint(self, manager: Any, *, version: str | None = None) -> Any:
        load_components = getattr(manager, "load_components", None)
        if not callable(load_components):
            raise TypeError("checkpoint manager must expose load_components()")
        baseline_model = self.state_dict()
        baseline_agent_state = self.agent_state()
        try:
            result = load_components(
                version,
                components=("model", "agent_state"),
                expected_codecs={"model": "numpy", "agent_state": "agent-state"},
            )
            components = getattr(result, "components", None)
            if not isinstance(components, Mapping):
                raise QNNInputError(
                    "checkpoint load result does not contain components"
                )
            self.load_state_dict(components["model"])
            self.load_agent_state(components["agent_state"])
        except Exception as exc:
            # The checkpoint manager decodes every component before returning,
            # while this facade owns its final in-memory application. Preserve
            # the same all-or-nothing behavior across the two local loads.
            self.load_state_dict(baseline_model)
            self.load_agent_state(baseline_agent_state)
            self._mark_runtime_degraded("persistence", "qnn.checkpoint.load", exc)
            raise
        self._mark_runtime_recovered("persistence", "qnn.checkpoint.load")
        return result

    def _latest_checkpoint_metrics(self) -> dict[str, float]:
        loss_values = tuple(self.performance_metrics.get("loss", ()))
        return {"loss": float(loss_values[-1])} if loss_values else {}

    def tuning_transaction_factory(
        self,
        agent_builder: Any,
        checkpoint_manager: Any = None,
    ) -> Any:
        """Return an opt-in factory accepted by ``src.tuning.AgentEvaluator``."""

        from .qnn.integration import make_qnn_transaction_factory

        return make_qnn_transaction_factory(
            agent_builder,
            checkpoint_manager=checkpoint_manager,
        )

    # ------------------------------------------------------------------
    # Optional visualization; no import-time plotting dependency
    # ------------------------------------------------------------------
    def visualize_output(
        self,
        output_sequence: Sequence[Any],
        *,
        show: bool = False,
    ) -> Any:
        if self.num_qubits == 1:
            return self.visualize_bloch(output_sequence, show=show)
        return self.visualize_amplitudes(output_sequence, show=show)

    def visualize_bloch(
        self,
        output_sequence: Sequence[Any],
        *,
        show: bool = False,
    ) -> Any:
        import matplotlib.pyplot as plt

        states = self._normalize_sequence(output_sequence, name="output_sequence")
        if self.num_qubits != 1:
            raise QNNInputError("Bloch visualization requires exactly one qubit")
        figure = plt.figure(figsize=(8, 6))
        axes = figure.add_subplot(111, projection="3d")
        for state in states:
            x, y, z = self._bloch_coordinates(state)
            axes.scatter(
                x,
                y,
                z, # type: ignore
                color="blue",
                s=50,
            )
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
    def _bloch_coordinates(state: Any) -> tuple[float, float, float]:
        amplitudes = np.asarray(state, dtype=np.complex128)
        if amplitudes.shape != (2,):
            raise QNNInputError("Bloch coordinates require a one-qubit state")
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

        states = self._normalize_sequence(output_sequence, name="output_sequence")
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


# Bounded import compatibility for the former monolithic module.
Task = QNNTask


__all__ = [
    "PerformanceEvaluator",
    "QNNAgent",
    "QNNConfig",
    "QNNTask",
    "QuantumGate",
    "Task",
]


if __name__ == "__main__":
    basis_zero = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    agent = QNNAgent(config={"num_qubits": 2, "num_quantum_layers": 1, "seed": 7})
    result = agent.perform_task({"mode": "infer", "input_sequences": [basis_zero]})
    print(
        {
            "status": result["status"],
            "model_kind": result["model_kind"],
            "norm_error": result["metrics"]["norm_error"],
        }
    )