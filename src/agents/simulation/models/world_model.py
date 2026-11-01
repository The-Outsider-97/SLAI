"""World-model adapters for mechanistic, learned, synthetic, and twin simulations.

Learned predictive rollouts are informed by Ha & Schmidhuber (2018) and Hafner
et al. (2020), but this module deliberately excludes policy learning/control.
Digital-model/shadow/twin distinctions follow Kritzinger et al. (2018) and Tao
et al. (2019); agent/environment representation is consistent with Macal &
North (2010).  External physical state is consumed through synchronization
callbacks; perception, SLAM, sensor fusion, and state estimation remain outside
Simulation.
"""

from __future__ import annotations

__version__ = "2.3.0"

import numpy as np

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ..simulation_types import TransitionOutcome
from ..utils.config_loader import get_config_section, load_global_config
from ..utils.simulation_errors import *
from ..utils.simulation_helpers import *
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("World Model")


class WorldModelKind(str, Enum):
    SIMULATION = "simulation"
    MECHANISTIC = "mechanistic"
    LEARNED_WORLD_MODEL = "learned_world_model"
    AGENT_BASED = "agent_based"
    DIGITAL_MODEL = "digital_model"
    DIGITAL_SHADOW = "digital_shadow"
    DIGITAL_TWIN = "digital_twin"


@dataclass(frozen=True, slots=True)
class WorldModelStatus:
    model_id: str
    version: str
    kind: WorldModelKind
    stochastic: bool
    synchronized: bool
    bidirectional_sync_capable: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_json_safe({
            "model_id": self.model_id,
            "version": self.version,
            "kind": self.kind.value,
            "stochastic": self.stochastic,
            "synchronized": self.synchronized,
            "bidirectional_sync_capable": self.bidirectional_sync_capable,
            "metadata": dict(self.metadata),
        })


class WorldModel:
    """Adapter around predictive callables or step-based synthetic environments."""

    thread_safe = False

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        slaienv: Optional[Any] = None,
        *,
        predictor: Optional[Callable[..., Any]] = None,
        initializer: Optional[Callable[..., Any]] = None,
        observer: Optional[Callable[..., Any]] = None,
        kind: WorldModelKind | str = WorldModelKind.SIMULATION,
        model_id: str = "world-model",
        version: str = __version__,
        stochastic: bool = False,
        external_to_digital: Optional[Callable[..., Any]] = None,
        digital_to_external: Optional[Callable[..., Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        global_config = load_global_config()
        section = dict(get_config_section("world_model", config=global_config) or {})
        if config:
            section.update(dict(config))
        self.env = slaienv
        self.predictor = predictor
        self.initializer = initializer
        self.observer = observer
        self.kind = WorldModelKind(kind)
        self.model_id = str(model_id or section.get("model_id", "world-model"))
        self.version = str(version)
        self.stochastic = bool(stochastic)
        self.external_to_digital = external_to_digital
        self.digital_to_external = digital_to_external
        self.metadata = dict(metadata or {})
        self._last_state: Any = None
        self._synchronized = False
        if not self.model_id.strip():
            raise SimulationValidationError("world-model model_id must be non-empty")
        if self.predictor is None and self.env is None:
            raise SimulationModelError("WorldModel requires either a predictor or a step-based environment")
        if self.kind is WorldModelKind.DIGITAL_SHADOW and self.external_to_digital is None:
            raise SimulationModelError("DIGITAL_SHADOW requires an external_to_digital synchronization callback")
        if self.kind is WorldModelKind.DIGITAL_TWIN and (
            self.external_to_digital is None or self.digital_to_external is None
        ):
            raise SimulationModelError(
                "DIGITAL_TWIN requires explicit bidirectional synchronization callbacks"
            )
        logger.info("World model initialized | model_id=%s kind=%s", self.model_id, self.kind.value)

    @staticmethod
    def _parse_reset(output: Any) -> tuple[Any, Mapping[str, Any]]:
        if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], Mapping):
            return output[0], dict(output[1])
        return output, {}

    @staticmethod
    def _parse_step(output: Any) -> tuple[Any, bool, Mapping[str, Any]]:
        if not isinstance(output, tuple):
            return output, False, {}
        if len(output) == 5:
            state, signal, terminated, truncated, info = output
            metadata = dict(info) if isinstance(info, Mapping) else {}
            metadata.setdefault("environment_signal", to_json_safe(signal))
            metadata.setdefault("truncated", bool(truncated))
            return state, bool(terminated or truncated), metadata
        if len(output) == 4:
            state, signal, done, info = output
            metadata = dict(info) if isinstance(info, Mapping) else {}
            metadata.setdefault("environment_signal", to_json_safe(signal))
            return state, bool(done), metadata
        if len(output) == 2 and isinstance(output[1], Mapping):
            return output[0], False, dict(output[1])
        return output[0] if output else None, False, {}

    def initialize(self, *, initial_state: Any, parameters: Mapping[str, Any], rng: np.random.Generator) -> Any:
        if self.initializer is not None:
            state = invoke_with_supported_kwargs(
                self.initializer,
                initial_state=clone_state(initial_state),
                parameters=dict(parameters),
                rng=rng,
            )
        elif self.env is not None and hasattr(self.env, "reset"):
            # Every rollout resets mutable environment state.  A derived seed is
            # consumed from the owned run RNG so environment reset remains fully
            # reproducible and recorded by the RNG-state snapshots.
            env_seed = int(rng.integers(0, 2**32, dtype=np.uint64))
            reset = invoke_with_supported_kwargs(getattr(self.env, "reset"), seed=env_seed)
            reset_state, _ = self._parse_reset(reset)
            if initial_state is None:
                state = reset_state
            else:
                setter = getattr(self.env, "set_state", None)
                if not callable(setter):
                    raise SimulationModelError(
                        "step-based environments require set_state(state) when an explicit initial_state is supplied; "
                        "otherwise the environment's hidden state would diverge from the trajectory state"
                    )
                invoke_with_supported_kwargs(setter, state=clone_state(initial_state))
                state = clone_state(initial_state)
        else:
            state = clone_state(initial_state)
        self._last_state = clone_state(state)
        return state

    def transition(
        self,
        *,
        state: Any,
        action: Any,
        interventions: tuple[Any, ...],
        parameters: Mapping[str, Any],
        rng: np.random.Generator,
        step: int,
        simulation_time: float,
        dt: float,
    ) -> TransitionOutcome:
        if self.predictor is not None:
            raw = invoke_with_supported_kwargs(
                self.predictor,
                state=state,
                action=action,
                interventions=interventions,
                parameters=dict(parameters),
                rng=rng,
                step=step,
                simulation_time=simulation_time,
                time=simulation_time,
                dt=dt,
            )
            outcome = raw if isinstance(raw, TransitionOutcome) else TransitionOutcome(state=raw)
        else:
            if self.env is None or not hasattr(self.env, "step"):
                raise SimulationModelError("configured environment does not expose step(action)")
            if interventions:
                apply_intervention = getattr(self.env, "apply_intervention", None)
                if not callable(apply_intervention):
                    raise SimulationModelError(
                        "environment cannot execute supplied interventions",
                        context={"intervention_count": len(interventions)},
                    )
                for intervention in interventions:
                    invoke_with_supported_kwargs(
                        apply_intervention,
                        intervention=intervention,
                        payload=getattr(intervention, "payload", intervention),
                    )
            raw = getattr(self.env, "step")(action)
            next_state, terminal, metadata = self._parse_step(raw)
            outcome = TransitionOutcome(state=next_state, terminal=terminal, metadata=metadata)
        self._last_state = clone_state(outcome.state)
        return outcome

    def observe(self, *, state: Any) -> Any:
        if self.observer is None:
            return clone_state(state)
        return invoke_with_supported_kwargs(self.observer, state=state)

    def sync_external_state(self, state: Any, *, metadata: Optional[Mapping[str, Any]] = None) -> Any:
        """Accept an already estimated external state; no perception is performed."""
        if self.external_to_digital is None:
            if self.kind in {WorldModelKind.DIGITAL_SHADOW, WorldModelKind.DIGITAL_TWIN}:
                raise SimulationModelError("external-to-digital synchronization is not configured")
            digital = clone_state(state)
        else:
            digital = invoke_with_supported_kwargs(
                self.external_to_digital,
                state=clone_state(state),
                metadata=dict(metadata or {}),
            )
        self._last_state = clone_state(digital)
        self._synchronized = True
        return clone_state(digital)

    def publish_digital_state(self, state: Any, *, metadata: Optional[Mapping[str, Any]] = None) -> Any:
        """Invoke an explicitly configured twin callback; never performs actuation itself."""
        if self.kind is not WorldModelKind.DIGITAL_TWIN or self.digital_to_external is None:
            raise SimulationModelError("bidirectional digital-twin synchronization is not available")
        return invoke_with_supported_kwargs(
            self.digital_to_external,
            state=clone_state(state),
            metadata=dict(metadata or {}),
        )

    def snapshot(self) -> Mapping[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "kind": self.kind.value,
            "stochastic": self.stochastic,
            "synchronized": self._synchronized,
            "bidirectional_sync_capable": self.external_to_digital is not None and self.digital_to_external is not None,
            "metadata": to_json_safe(self.metadata),
        }

    def status(self) -> WorldModelStatus:
        return WorldModelStatus(
            model_id=self.model_id,
            version=self.version,
            kind=self.kind,
            stochastic=self.stochastic,
            synchronized=self._synchronized,
            bidirectional_sync_capable=self.external_to_digital is not None and self.digital_to_external is not None,
            metadata=self.metadata,
        )


__all__ = [
    "WorldModel", 
    "WorldModelKind", 
    "WorldModelStatus"
]
