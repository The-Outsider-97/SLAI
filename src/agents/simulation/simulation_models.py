"""Model contracts and registry for the SLAI Simulation subsystem.

The model/simulator separation follows Zeigler, Praehofer & Kim (2000) and
Banks et al. (2010): models expose state-evolution semantics, while engines own
execution clocks, horizons, trajectories, and run lifecycle.  This module does
not select actions or optimize model parameters.
"""

from __future__ import annotations

__version__ = "2.3.0"

import threading

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

from .simulation_types import TransitionOutcome
from .utils.config_loader import get_config_section, load_global_config
from .utils.simulation_errors import *
from .utils.simulation_helpers import *
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Simulation Models")


@runtime_checkable
class SimulationModel(Protocol):
    """Runtime protocol for a state-transition simulation model."""

    model_id: str
    version: str
    stochastic: bool

    def initialize(self, *, initial_state: Any, parameters: Mapping[str, Any], rng: Any) -> Any:
        """Return the initialized state for one independent simulation run."""
        ...

    def transition(
        self,
        *,
        state: Any,
        action: Any,
        interventions: tuple[Any, ...],
        parameters: Mapping[str, Any],
        rng: Any,
        step: int,
        simulation_time: float,
        dt: float,
    ) -> Any | TransitionOutcome:
        """Propagate one state transition under externally supplied inputs."""
        ...

    def observe(self, *, state: Any) -> Any:
        """Return a model-specific observation without choosing an action."""
        ...

    def snapshot(self) -> Mapping[str, Any]:
        """Return reproducibility-relevant model metadata/state."""
        ...


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    model_id: str
    version: str
    stochastic: bool
    model_type: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "stochastic": self.stochastic,
            "model_type": self.model_type,
            "metadata": to_json_safe(dict(self.metadata)),
        }


class CallableSimulationModel:
    """Adapter for narrowly scoped transition callables.

    Integration functions receive only arguments they explicitly accept.  The
    adapter intentionally contains no planner, reward maximizer, or policy.
    """

    def __init__(
        self,
        transition_fn: Callable[..., Any],
        *,
        model_id: str = "callable-model",
        version: str = "1",
        stochastic: bool = False,
        initialize_fn: Optional[Callable[..., Any]] = None,
        observe_fn: Optional[Callable[..., Any]] = None,
        snapshot_fn: Optional[Callable[..., Mapping[str, Any]]] = None,
        thread_safe: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not callable(transition_fn):
            raise SimulationValidationError("transition_fn must be callable")
        if not str(model_id).strip():
            raise SimulationValidationError("model_id must be non-empty")
        self.model_id = str(model_id)
        self.version = str(version)
        self.stochastic = bool(stochastic)
        self.thread_safe = bool(thread_safe)
        self._transition_fn = transition_fn
        self._initialize_fn = initialize_fn
        self._observe_fn = observe_fn
        self._snapshot_fn = snapshot_fn
        self.metadata = dict(metadata or {})

    def initialize(self, *, initial_state: Any, parameters: Mapping[str, Any], rng: Any) -> Any:
        if self._initialize_fn is None:
            return clone_state(initial_state)
        return invoke_with_supported_kwargs(
            self._initialize_fn,
            initial_state=clone_state(initial_state),
            state=clone_state(initial_state),
            parameters=dict(parameters),
            rng=rng,
        )

    def transition(
        self,
        *,
        state: Any,
        action: Any,
        interventions: tuple[Any, ...],
        parameters: Mapping[str, Any],
        rng: Any,
        step: int,
        simulation_time: float,
        dt: float,
    ) -> Any | TransitionOutcome:
        return invoke_with_supported_kwargs(
            self._transition_fn,
            state=state,
            action=action,
            interventions=interventions,
            intervention=interventions[0] if len(interventions) == 1 else interventions,
            parameters=dict(parameters),
            rng=rng,
            step=step,
            simulation_time=simulation_time,
            time=simulation_time,
            dt=dt,
        )

    def observe(self, *, state: Any) -> Any:
        if self._observe_fn is None:
            return clone_state(state)
        return invoke_with_supported_kwargs(self._observe_fn, state=state)

    def snapshot(self) -> Mapping[str, Any]:
        if self._snapshot_fn is not None:
            payload = invoke_with_supported_kwargs(self._snapshot_fn)
            if not isinstance(payload, Mapping):
                raise SimulationModelError("snapshot_fn must return a mapping")
            return dict(payload)
        return {
            "model_id": self.model_id,
            "version": self.version,
            "stochastic": self.stochastic,
            "thread_safe": self.thread_safe,
            "model_type": type(self).__name__,
            "metadata": to_json_safe(self.metadata),
        }


class SimulationModelRegistry:
    """Thread-safe registry of explicitly registered simulation models."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        global_config = load_global_config()
        section = dict(get_config_section("simulation_model", config=global_config) or {})
        if config:
            section.update(dict(config))
        self.allow_replace = bool(section.get("allow_replace", False))
        self._models: dict[str, SimulationModel] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _key(model_id: str) -> str:
        key = str(model_id).strip()
        if not key:
            raise SimulationValidationError("model_id must be non-empty")
        return key

    def register(self, model: SimulationModel, *, replace: Optional[bool] = None) -> ModelDescriptor:
        if not hasattr(model, "transition") or not callable(getattr(model, "transition")):
            raise SimulationModelError("model must expose a callable transition method")
        model_id = self._key(getattr(model, "model_id", ""))
        should_replace = self.allow_replace if replace is None else bool(replace)
        with self._lock:
            if model_id in self._models and not should_replace:
                raise SimulationModelError(
                    "simulation model is already registered",
                    context={"model_id": model_id},
                )
            self._models[model_id] = model
        descriptor = self.describe(model_id)
        logger.info("Simulation model registered | model_id=%s version=%s", descriptor.model_id, descriptor.version)
        return descriptor

    def get(self, model_id: str) -> SimulationModel:
        key = self._key(model_id)
        with self._lock:
            try:
                return self._models[key]
            except KeyError as exc:
                raise SimulationModelError(
                    "simulation model is not registered",
                    context={"model_id": key},
                    cause=exc,
                ) from exc

    def unregister(self, model_id: str) -> Optional[SimulationModel]:
        key = self._key(model_id)
        with self._lock:
            return self._models.pop(key, None)

    def describe(self, model_id: str) -> ModelDescriptor:
        model = self.get(model_id)
        metadata: Mapping[str, Any] = {}
        snapshot = getattr(model, "snapshot", None)
        if callable(snapshot):
            try:
                candidate = snapshot()
                if isinstance(candidate, Mapping):
                    metadata = candidate
            except Exception as exc:
                logger.debug("Model snapshot unavailable | model_id=%s error=%s", model_id, exc)
        return ModelDescriptor(
            model_id=str(getattr(model, "model_id", model_id)),
            version=str(getattr(model, "version", "unknown")),
            stochastic=bool(getattr(model, "stochastic", False)),
            model_type=type(model).__name__,
            metadata=metadata,
        )

    def descriptors(self) -> tuple[ModelDescriptor, ...]:
        with self._lock:
            keys = tuple(sorted(self._models))
        return tuple(self.describe(key) for key in keys)

    def __contains__(self, model_id: object) -> bool:
        if not isinstance(model_id, str):
            return False
        with self._lock:
            return model_id in self._models

    def __len__(self) -> int:
        with self._lock:
            return len(self._models)


__all__ = [
    "CallableSimulationModel",
    "ModelDescriptor",
    "SimulationModel",
    "SimulationModelRegistry",
]
