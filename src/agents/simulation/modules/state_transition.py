"""Validated one-step state propagation for SLAI Simulation.

The transition contract follows Zeigler et al. (2000) for state-transition
systems and Puterman (1994) for stochastic kernels, without importing policy
optimization.  Externally supplied actions and interventions are propagated;
this module never chooses them.
"""

from __future__ import annotations

__version__ = "2.3.0"

import numpy as np

from collections.abc import Callable, Mapping
from typing import Any, Optional

from ..simulation_types import TransitionOutcome
from ..utils.config_loader import get_config_section, load_global_config
from ..utils.simulation_errors import *
from ..utils.simulation_helpers import *
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("State Transition")


class StateTransition:
    """Apply one bounded, validated transition to a supplied model/callable."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        global_config = load_global_config()
        section = dict(get_config_section("state_transition", config=global_config) or {})
        if config:
            section.update(dict(config))
        self.max_state_size = int(section.get("max_state_size", 1_000_000))
        self.clone_input_state = bool(section.get("clone_input_state", True))
        self.clone_output_state = bool(section.get("clone_output_state", True))
        if self.max_state_size <= 0:
            raise SimulationValidationError("state_transition.max_state_size must be > 0")

    def apply(
        self,
        model: Any,
        *,
        state: Any,
        action: Any,
        interventions: tuple[Any, ...] = (),
        parameters: Optional[Mapping[str, Any]] = None,
        rng: np.random.Generator,
        step: int,
        simulation_time: float,
        dt: float,
    ) -> TransitionOutcome:
        if not isinstance(rng, np.random.Generator):
            raise SimulationValidationError("rng must be numpy.random.Generator")
        if step < 0 or simulation_time < 0.0 or dt <= 0.0:
            raise SimulationValidationError(
                "transition step/time must be non-negative and dt must be positive",
                context={"step": step, "simulation_time": simulation_time, "dt": dt},
            )
        validate_state(state, max_state_size=self.max_state_size)
        input_state = clone_state(state) if self.clone_input_state else state
        transition: Optional[Callable[..., Any]]
        if hasattr(model, "transition") and callable(getattr(model, "transition")):
            transition = getattr(model, "transition")
        elif callable(model):
            transition = model
        else:
            raise SimulationModelError("model must be callable or expose transition()")
        assert transition is not None
        try:
            raw = invoke_with_supported_kwargs(
                transition,
                state=input_state,
                action=action,
                interventions=interventions,
                intervention=interventions[0] if len(interventions) == 1 else interventions,
                parameters=dict(parameters or {}),
                rng=rng,
                step=int(step),
                simulation_time=float(simulation_time),
                time=float(simulation_time),
                dt=float(dt),
            )
        except SimulationError:
            raise
        except Exception as exc:
            raise SimulationTransitionError(
                "state transition callable failed",
                cause=exc,
                context={
                    "step": step,
                    "simulation_time": simulation_time,
                    "model_type": type(model).__name__,
                },
                operation="transition",
            ) from exc
        outcome = raw if isinstance(raw, TransitionOutcome) else TransitionOutcome(state=raw)
        validate_state(outcome.state, max_state_size=self.max_state_size)
        output_state = clone_state(outcome.state) if self.clone_output_state else outcome.state
        return TransitionOutcome(
            state=output_state,
            terminal=bool(outcome.terminal),
            metadata=dict(outcome.metadata),
        )

    __call__ = apply


__all__ = ["StateTransition"]
