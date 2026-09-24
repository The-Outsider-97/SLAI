"""Typed contracts and formal vocabulary for the SLAI Simulation subsystem.

The records in this module describe simulation inputs and descriptive outputs;
they do not encode preferences, utilities, policies, or plan-selection logic.
State-transition semantics are grounded in Zeigler, Praehofer & Kim (2000) and
Puterman (1994).  Stochastic runs retain explicit seeds and RNG metadata for
reproducibility in the sense advocated by Sandve et al. (2013).
"""

from __future__ import annotations

__version__ = "2.3.0"

import math
import uuid

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .utils.simulation_errors import SimulationValidationError
from .utils.simulation_helpers import clone_state, to_json_safe
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("Simulation Types")

SimulationState = Any
Action = Any
TransitionCallable = Callable[..., Any]
TerminationPredicate = Callable[[Any, int, float], bool]
CancellationCheck = Callable[[], bool]


class SimulationMode(str, Enum):
    DETERMINISTIC = "deterministic"
    STOCHASTIC = "stochastic"


class TerminationReason(str, Enum):
    ACTIONS_EXHAUSTED = "actions_exhausted"
    HORIZON_STEPS = "horizon_steps"
    HORIZON_TIME = "horizon_time"
    MODEL_TERMINATED = "model_terminated"
    USER_CONDITION = "user_condition"
    CANCELLED = "cancelled"
    ENGINE_LIMIT = "engine_limit"


class SamplingStrategy(str, Enum):
    RANDOM = "random"
    LATIN_HYPERCUBE = "latin_hypercube"


@dataclass(frozen=True, slots=True)
class SimulationSeed:
    value: int
    stream: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int) or self.value < 0:
            raise SimulationValidationError("SimulationSeed.value must be a non-negative integer")
        if isinstance(self.stream, bool) or not isinstance(self.stream, int) or self.stream < 0:
            raise SimulationValidationError("SimulationSeed.stream must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class Intervention:
    """Externally defined state-transition intervention.

    Simulation executes the supplied intervention; it does not infer whether the
    intervention is causal, valid, desirable, or safe.
    """

    payload: Any
    intervention_id: str = field(default_factory=lambda: f"intervention-{uuid.uuid4().hex[:16]}")
    step: Optional[int] = None
    time: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.intervention_id).strip():
            raise SimulationValidationError("intervention_id must be non-empty")
        if self.step is not None:
            if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
                raise SimulationValidationError("intervention step must be an integer >= 0")
        if self.time is not None and (not math.isfinite(float(self.time)) or float(self.time) < 0.0):
            raise SimulationValidationError("intervention time must be finite and >= 0")

    def applies(self, step: int, time_value: float, *, tolerance: float = 1.0e-12) -> bool:
        if self.step is not None and int(self.step) != int(step):
            return False
        if self.time is not None and abs(float(self.time) - float(time_value)) > tolerance:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "intervention_id": self.intervention_id,
            "step": self.step,
            "time": self.time,
            "payload": to_json_safe(self.payload),
            "metadata": to_json_safe(dict(self.metadata)),
        }


@dataclass(frozen=True, slots=True)
class Perturbation:
    parameter: str
    value: Any
    perturbation_id: str = field(default_factory=lambda: f"perturbation-{uuid.uuid4().hex[:16]}")
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.parameter).strip():
            raise SimulationValidationError("perturbation parameter must be non-empty")
        if not str(self.perturbation_id).strip():
            raise SimulationValidationError("perturbation_id must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "perturbation_id": self.perturbation_id,
            "parameter": self.parameter,
            "value": to_json_safe(self.value),
            "metadata": to_json_safe(dict(self.metadata)),
        }


@dataclass(frozen=True, slots=True)
class TransitionOutcome:
    state: Any
    terminal: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    index: int
    simulation_time: float
    state: Any
    action: Any = None
    interventions: tuple[Intervention, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or self.index < 0:
            raise SimulationValidationError("trajectory step index must be >= 0")
        if not math.isfinite(float(self.simulation_time)) or float(self.simulation_time) < 0.0:
            raise SimulationValidationError("trajectory simulation_time must be finite and >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "simulation_time": float(self.simulation_time),
            "state": to_json_safe(self.state),
            "action": to_json_safe(self.action),
            "interventions": [item.to_dict() for item in self.interventions],
            "metadata": to_json_safe(dict(self.metadata)),
        }


@dataclass(frozen=True, slots=True)
class Trajectory:
    trajectory_id: str
    steps: tuple[TrajectoryStep, ...]
    termination_reason: TerminationReason
    seed: int
    model_id: str
    scenario_id: Optional[str] = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.steps:
            raise SimulationValidationError("trajectory must contain at least the initial state")
        if self.steps[0].index != 0:
            raise SimulationValidationError("trajectory must start at step index 0")
        if not str(self.trajectory_id).strip() or not str(self.model_id).strip():
            raise SimulationValidationError("trajectory_id and model_id must be non-empty")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0 or self.seed > (1 << 64) - 1:
            raise SimulationValidationError("trajectory seed must be in [0, 2**64 - 1]")
        previous_time = -math.inf
        for expected_index, step in enumerate(self.steps):
            if step.index != expected_index:
                raise SimulationValidationError("trajectory step indices must be contiguous")
            if step.simulation_time < previous_time:
                raise SimulationValidationError("trajectory simulation_time must be monotonic")
            previous_time = step.simulation_time

    @property
    def initial_state(self) -> Any:
        return clone_state(self.steps[0].state)

    @property
    def terminal_state(self) -> Any:
        return clone_state(self.steps[-1].state)

    @property
    def transition_count(self) -> int:
        return max(0, len(self.steps) - 1)

    @property
    def duration(self) -> float:
        return float(self.steps[-1].simulation_time - self.steps[0].simulation_time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "scenario_id": self.scenario_id,
            "model_id": self.model_id,
            "seed": self.seed,
            "termination_reason": self.termination_reason.value,
            "transition_count": self.transition_count,
            "duration": self.duration,
            "steps": [step.to_dict() for step in self.steps],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class SimulationRuntimeMetadata:
    started_at: str
    finished_at: str
    duration_seconds: float
    transition_count: int
    deterministic: bool
    numpy_version: str
    python_version: str
    rng_bit_generator: str

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.duration_seconds)) or self.duration_seconds < 0.0:
            raise SimulationValidationError("runtime duration_seconds must be finite and >= 0")
        if isinstance(self.transition_count, bool) or self.transition_count < 0:
            raise SimulationValidationError("runtime transition_count must be >= 0")
        if not self.rng_bit_generator:
            raise SimulationValidationError("runtime rng_bit_generator must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "transition_count": self.transition_count,
            "deterministic": self.deterministic,
            "numpy_version": self.numpy_version,
            "python_version": self.python_version,
            "rng_bit_generator": self.rng_bit_generator,
        }


@dataclass(frozen=True, slots=True)
class DistributionSummary:
    count: int
    mean: float
    variance: float
    standard_deviation: float
    minimum: float
    maximum: float
    quantiles: Mapping[str, float]
    confidence_interval: Optional[tuple[float, float]] = None

    def to_dict(self) -> dict[str, Any]:
        return to_json_safe({
            "count": self.count,
            "mean": self.mean,
            "variance": self.variance,
            "standard_deviation": self.standard_deviation,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "quantiles": dict(self.quantiles),
            "confidence_interval": self.confidence_interval,
        })


@dataclass(frozen=True, slots=True)
class ConvergenceDiagnostics:
    sample_count: int
    cumulative_mean: float
    standard_error: float
    relative_standard_error: Optional[float]
    recent_mean_shift: Optional[float]
    converged: bool
    tolerance: float

    def to_dict(self) -> dict[str, Any]:
        return to_json_safe({
            "sample_count": self.sample_count,
            "cumulative_mean": self.cumulative_mean,
            "standard_error": self.standard_error,
            "relative_standard_error": self.relative_standard_error,
            "recent_mean_shift": self.recent_mean_shift,
            "converged": self.converged,
            "tolerance": self.tolerance,
        })


@dataclass(frozen=True, slots=True)
class SensitivityEstimate:
    method: str
    first_order: Mapping[str, float] = field(default_factory=dict)
    total_order: Mapping[str, float] = field(default_factory=dict)
    elementary_mean: Mapping[str, float] = field(default_factory=dict)
    elementary_mean_abs: Mapping[str, float] = field(default_factory=dict)
    elementary_std: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_json_safe({
            "method": self.method,
            "first_order": dict(self.first_order),
            "total_order": dict(self.total_order),
            "elementary_mean": dict(self.elementary_mean),
            "elementary_mean_abs": dict(self.elementary_mean_abs),
            "elementary_std": dict(self.elementary_std),
            "metadata": dict(self.metadata),
        })


@dataclass(frozen=True, slots=True)
class Scenario:
    scenario_id: str
    parent_scenario_id: Optional[str] = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    interventions: tuple[Intervention, ...] = ()
    perturbations: tuple[Perturbation, ...] = ()
    seed: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.scenario_id).strip():
            raise SimulationValidationError("scenario_id must be non-empty")
        if self.parent_scenario_id == self.scenario_id:
            raise SimulationValidationError("scenario cannot be its own parent")
        if not isinstance(self.parameters, Mapping) or not isinstance(self.metadata, Mapping):
            raise SimulationValidationError("scenario parameters/metadata must be mappings")
        if any(not isinstance(item, Intervention) for item in self.interventions):
            raise SimulationValidationError("scenario interventions must be Intervention instances")
        if any(not isinstance(item, Perturbation) for item in self.perturbations):
            raise SimulationValidationError("scenario perturbations must be Perturbation instances")
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0 or self.seed > (1 << 64) - 1):
            raise SimulationValidationError("scenario seed must be in [0, 2**64 - 1]")
        object.__setattr__(self, "interventions", tuple(self.interventions))
        object.__setattr__(self, "perturbations", tuple(self.perturbations))

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "parent_scenario_id": self.parent_scenario_id,
            "parameters": to_json_safe(dict(self.parameters)),
            "interventions": [item.to_dict() for item in self.interventions],
            "perturbations": [item.to_dict() for item in self.perturbations],
            "seed": self.seed,
            "metadata": to_json_safe(dict(self.metadata)),
        }


@dataclass(frozen=True, slots=True)
class ScenarioBranch:
    scenario_id: str
    parent_scenario_id: Optional[str]
    depth: int
    child_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_json_safe({
            "scenario_id": self.scenario_id,
            "parent_scenario_id": self.parent_scenario_id,
            "depth": self.depth,
            "child_ids": list(self.child_ids),
            "metadata": dict(self.metadata),
        })


@dataclass(frozen=True, slots=True)
class SimulationRequest:
    initial_state: Any
    actions: Sequence[Any]
    parameters: Mapping[str, Any] = field(default_factory=dict)
    interventions: Sequence[Intervention] = field(default_factory=tuple)
    run_id: str = field(default_factory=lambda: f"sim-{uuid.uuid4().hex[:20]}")
    scenario_id: Optional[str] = None
    seed: Optional[int] = None
    mode: SimulationMode = SimulationMode.DETERMINISTIC
    time_step: float = 1.0
    horizon_steps: Optional[int] = None
    horizon_time: Optional[float] = None
    timeout_seconds: Optional[float] = None
    termination_condition: Optional[TerminationPredicate] = field(default=None, repr=False, compare=False)
    cancellation_check: Optional[CancellationCheck] = field(default=None, repr=False, compare=False)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.run_id).strip():
            raise SimulationValidationError("run_id must be non-empty")
        if not isinstance(self.actions, Sequence) or isinstance(self.actions, (str, bytes, bytearray)):
            raise SimulationValidationError("actions must be an ordered sequence")
        if not math.isfinite(float(self.time_step)) or float(self.time_step) <= 0.0:
            raise SimulationValidationError("time_step must be finite and > 0")
        if self.horizon_steps is not None and (isinstance(self.horizon_steps, bool) or int(self.horizon_steps) < 0):
            raise SimulationValidationError("horizon_steps must be >= 0")
        if self.horizon_time is not None and (not math.isfinite(float(self.horizon_time)) or float(self.horizon_time) < 0.0):
            raise SimulationValidationError("horizon_time must be finite and >= 0")
        if self.timeout_seconds is not None and (not math.isfinite(float(self.timeout_seconds)) or float(self.timeout_seconds) <= 0.0):
            raise SimulationValidationError("timeout_seconds must be finite and > 0")
        if not isinstance(self.parameters, Mapping) or not isinstance(self.metadata, Mapping):
            raise SimulationValidationError("request parameters/metadata must be mappings")
        if any(not isinstance(item, Intervention) for item in self.interventions):
            raise SimulationValidationError("request interventions must be Intervention instances")
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0 or self.seed > (1 << 64) - 1):
            raise SimulationValidationError("seed must be in [0, 2**64 - 1]")
        if self.termination_condition is not None and not callable(self.termination_condition):
            raise SimulationValidationError("termination_condition must be callable")
        if self.cancellation_check is not None and not callable(self.cancellation_check):
            raise SimulationValidationError("cancellation_check must be callable")
        object.__setattr__(self, "mode", SimulationMode(self.mode))
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(self, "interventions", tuple(self.interventions))

    @property
    def deterministic(self) -> bool:
        return self.mode is SimulationMode.DETERMINISTIC

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "initial_state": to_json_safe(self.initial_state),
            "actions": to_json_safe(list(self.actions)),
            "parameters": to_json_safe(dict(self.parameters)),
            "interventions": [item.to_dict() for item in self.interventions],
            "seed": self.seed,
            "mode": self.mode.value,
            "time_step": self.time_step,
            "horizon_steps": self.horizon_steps,
            "horizon_time": self.horizon_time,
            "timeout_seconds": self.timeout_seconds,
            "has_termination_condition": self.termination_condition is not None,
            "has_cancellation_check": self.cancellation_check is not None,
            "metadata": to_json_safe(dict(self.metadata)),
        }


@dataclass(frozen=True, slots=True)
class SimulationResult:
    run_id: str
    scenario_id: Optional[str]
    model_id: str
    model_version: str
    trajectory: Trajectory
    actions_supplied: tuple[Any, ...]
    interventions_supplied: tuple[Intervention, ...]
    parameters: Mapping[str, Any]
    seed: int
    rng_state_initial: Mapping[str, Any]
    rng_state_final: Mapping[str, Any]
    runtime: SimulationRuntimeMetadata
    warnings: tuple[str, ...] = ()
    distributions: Mapping[str, DistributionSummary] = field(default_factory=dict)
    uncertainty: Mapping[str, Any] = field(default_factory=dict)
    sensitivity: Mapping[str, SensitivityEstimate] = field(default_factory=dict)
    branch_metadata: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def initial_state(self) -> Any:
        return self.trajectory.initial_state

    @property
    def terminal_state(self) -> Any:
        return self.trajectory.terminal_state

    @property
    def termination_reason(self) -> TerminationReason:
        return self.trajectory.termination_reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "initial_state": to_json_safe(self.initial_state),
            "terminal_state": to_json_safe(self.terminal_state),
            "trajectory": self.trajectory.to_dict(),
            "actions_supplied": to_json_safe(list(self.actions_supplied)),
            "interventions_supplied": [item.to_dict() for item in self.interventions_supplied],
            "parameters": to_json_safe(dict(self.parameters)),
            "seed": self.seed,
            "rng_state_initial": to_json_safe(dict(self.rng_state_initial)),
            "rng_state_final": to_json_safe(dict(self.rng_state_final)),
            "termination_reason": self.termination_reason.value,
            "warnings": list(self.warnings),
            "distributions": {name: value.to_dict() for name, value in self.distributions.items()},
            "uncertainty": to_json_safe(dict(self.uncertainty)),
            "sensitivity": {name: value.to_dict() for name, value in self.sensitivity.items()},
            "branch_metadata": to_json_safe(dict(self.branch_metadata)),
            "runtime": self.runtime.to_dict(),
            "metadata": to_json_safe(dict(self.metadata)),
        }


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    batch_id: str
    root_seed: int
    sample_seeds: tuple[int, ...]
    results: tuple[SimulationResult, ...]
    failures: tuple[Mapping[str, Any], ...] = ()
    summaries: Mapping[str, DistributionSummary] = field(default_factory=dict)
    convergence: Mapping[str, ConvergenceDiagnostics] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        observed = len(self.results) + len(self.failures)
        if len(self.sample_seeds) != observed:
            raise SimulationValidationError(
                "MonteCarloResult sample_seeds must match completed + failed samples",
                context={"sample_seeds": len(self.sample_seeds), "observed": observed},
            )

    @property
    def sample_count(self) -> int:
        return len(self.results) + len(self.failures)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "root_seed": self.root_seed,
            "sample_seeds": list(self.sample_seeds),
            "sample_count": self.sample_count,
            "results": [result.to_dict() for result in self.results],
            "failures": to_json_safe(list(self.failures)),
            "summaries": {name: value.to_dict() for name, value in self.summaries.items()},
            "convergence": {name: value.to_dict() for name, value in self.convergence.items()},
            "metadata": to_json_safe(dict(self.metadata)),
        }


@dataclass(frozen=True, slots=True)
class ScenarioBatchResult:
    batch_id: str
    results: Mapping[str, SimulationResult]
    branches: Mapping[str, ScenarioBranch]
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "results": {key: value.to_dict() for key, value in self.results.items()},
            "branches": {key: value.to_dict() for key, value in self.branches.items()},
            "warnings": list(self.warnings),
            "metadata": to_json_safe(dict(self.metadata)),
        }


@dataclass(frozen=True, slots=True)
class SimulationVocab:
    """Stable names for common Simulation contracts.

    Kept for compatibility with the current SimulationAgent constructor, which
    accepts ``SimulationVocab`` as its vocabulary object.
    """

    SimulationState: str = "SimulationState"
    Transition: str = "Transition"
    TransitionKernel: str = "TransitionKernel"
    Trajectory: str = "Trajectory"
    TrajectoryStep: str = "TrajectoryStep"
    SimulationRequest: str = "SimulationRequest"
    SimulationResult: str = "SimulationResult"
    SimulationRun: str = "SimulationRun"
    SimulationSeed: str = "SimulationSeed"
    Intervention: str = "Intervention"
    Perturbation: str = "Perturbation"
    Scenario: str = "Scenario"
    ScenarioBranch: str = "ScenarioBranch"
    EnvironmentState: str = "EnvironmentState"
    ModelParameters: str = "ModelParameters"
    StochasticParameters: str = "StochasticParameters"
    SimulationStatistics: str = "SimulationStatistics"
    TerminationCondition: str = "TerminationCondition"


class SimulationTypes:
    """Compatibility facade exposing the typed contract classes."""

    vocab = SimulationVocab()
    request_type = SimulationRequest
    result_type = SimulationResult
    trajectory_type = Trajectory
    scenario_type = Scenario

    def __init__(self, *_: Any, **__: Any) -> None:
        logger.debug("SimulationTypes initialized")


__all__ = [
    "Action",
    "CancellationCheck",
    "ConvergenceDiagnostics",
    "DistributionSummary",
    "Intervention",
    "MonteCarloResult",
    "Perturbation",
    "SamplingStrategy",
    "Scenario",
    "ScenarioBatchResult",
    "ScenarioBranch",
    "SensitivityEstimate",
    "SimulationMode",
    "SimulationRequest",
    "SimulationResult",
    "SimulationRuntimeMetadata",
    "SimulationSeed",
    "SimulationState",
    "SimulationTypes",
    "SimulationVocab",
    "TerminationPredicate",
    "TerminationReason",
    "Trajectory",
    "TrajectoryStep",
    "TransitionCallable",
    "TransitionOutcome",
]
