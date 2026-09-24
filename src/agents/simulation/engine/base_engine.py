"""Generic bounded execution semantics for SLAI Simulation models.

The engine implements the model/simulator separation described by Zeigler et
al. (2000), with discrete execution concepts consistent with Banks et al.
(2010) and run-design principles from Law (2015).  It exposes descriptive run
metadata for downstream verification/evaluation in the spirit of Sargent
(2013), but performs neither formal verification nor evaluative judgement.
"""

from __future__ import annotations

__version__ = "2.3.0"

import sys
import time
import uuid
import numpy as np

from collections.abc import Mapping
from typing import Any, Optional

from ..modules.state_transition import StateTransition
from ..simulation_types import *
from ..utils.config_loader import get_config_section, load_global_config
from ..utils.simulation_errors import *
from ..utils.simulation_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Simulation Base Engine")
printer = PrettyPrinter()


class BaseEngine:
    """Execute externally supplied actions through a supplied transition model."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None, *, transition: Optional[StateTransition] = None) -> None:
        global_config = load_global_config()
        section = dict(get_config_section("base_engine", config=global_config) or {})
        if config:
            section.update(dict(config))
        self.max_rollout_depth = int(section.get("max_rollout_depth", 10_000))
        self.max_state_size = int(section.get("max_state_size", 1_000_000))
        self.pretty_progress = bool(section.get("pretty_progress", False))
        timeout = section.get("timeout_seconds", None)
        self.default_timeout = None if timeout in (None, "", "none", "None") else float(timeout)
        if self.max_rollout_depth <= 0 or self.max_state_size <= 0:
            raise SimulationValidationError("engine rollout/state bounds must be > 0")
        if self.default_timeout is not None and self.default_timeout <= 0.0:
            raise SimulationValidationError("base_engine.timeout_seconds must be > 0")
        transition_config = dict(section.get("state_transition", {}) or {})
        transition_config.setdefault("max_state_size", self.max_state_size)
        self.transition = transition or StateTransition(transition_config)

    @staticmethod
    def _model_id(model: Any) -> str:
        return str(getattr(model, "model_id", type(model).__name__))

    @staticmethod
    def _model_version(model: Any) -> str:
        return str(getattr(model, "version", "unknown"))

    @staticmethod
    def _active_interventions(request: SimulationRequest, step: int, simulation_time: float) -> tuple[Any, ...]:
        return tuple(
            intervention
            for intervention in request.interventions
            if intervention.applies(step, simulation_time)
        )

    @staticmethod
    def _call_run_predicate(callback: Any, *, name: str, state: Any, step: int, simulation_time: float) -> bool:
        if callback is None:
            return False
        try:
            if name == "cancellation_check":
                return bool(callback())
            return bool(callback(state, step, simulation_time))
        except SimulationError:
            raise
        except Exception as exc:
            raise SimulationCallbackError(
                f"{name} failed during simulation",
                cause=exc,
                context={"step": step, "simulation_time": simulation_time},
                operation=name,
            ) from exc

    def _initialize_state(self, model: Any, request: SimulationRequest, rng: np.random.Generator) -> Any:
        initializer = getattr(model, "initialize", None)
        if callable(initializer):
            try:
                state = invoke_with_supported_kwargs(
                    initializer,
                    initial_state=clone_state(request.initial_state),
                    parameters=dict(request.parameters),
                    rng=rng,
                )
            except SimulationError:
                raise
            except Exception as exc:
                raise SimulationModelError(
                    "simulation model initialization failed",
                    cause=exc,
                    context={"model_id": self._model_id(model), "run_id": request.run_id},
                    operation="initialize",
                ) from exc
        else:
            state = clone_state(request.initial_state)
        validate_state(state, max_state_size=self.max_state_size)
        return state

    def run(
        self,
        request: SimulationRequest,
        model: Any,
        *,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        branch_metadata: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> SimulationResult:
        if not isinstance(request, SimulationRequest):
            raise SimulationValidationError("request must be SimulationRequest")
        if model is None:
            raise SimulationModelError("model must not be None")
        if rng is None:
            selected_seed = request.seed if seed is None else seed
            generator, concrete_seed = make_rng(selected_seed)
        else:
            selected_seed = request.seed if seed is None else seed
            if selected_seed is None:
                raise SimulationValidationError(
                    "an injected RNG requires its originating seed for reproducibility"
                )
            generator = rng
            concrete_seed = int(selected_seed)
        initial_rng_state = rng_state_snapshot(generator)
        timeout = request.timeout_seconds if request.timeout_seconds is not None else self.default_timeout
        started_monotonic = time.monotonic()
        started_at = utc_now_iso()
        model_id = self._model_id(model)
        model_version = self._model_version(model)
        warnings: list[str] = []
        if request.deterministic and bool(getattr(model, "stochastic", False)):
            raise SimulationValidationError(
                "stochastic models require SimulationMode.STOCHASTIC",
                context={"model_id": model_id, "run_id": request.run_id},
            )
        logger.info(
            "Simulation start | run_id=%s scenario=%s model=%s mode=%s seed=%s",
            request.run_id,
            request.scenario_id,
            model_id,
            request.mode.value,
            concrete_seed,
        )
        if self.pretty_progress:
            printer.status("START", f"Simulation {request.run_id} started", "info")

        state = self._initialize_state(model, request, generator)
        steps: list[TrajectoryStep] = [
            TrajectoryStep(index=0, simulation_time=0.0, state=clone_state(state))
        ]
        simulation_time = 0.0
        termination = TerminationReason.ACTIONS_EXHAUSTED

        if self._call_run_predicate(request.cancellation_check, name="cancellation_check", state=state, step=0, simulation_time=simulation_time):
            termination = TerminationReason.CANCELLED
        elif self._call_run_predicate(request.termination_condition, name="termination_condition", state=state, step=0, simulation_time=simulation_time):
            termination = TerminationReason.USER_CONDITION
        elif request.horizon_steps == 0:
            termination = TerminationReason.HORIZON_STEPS
        elif request.horizon_time == 0.0:
            termination = TerminationReason.HORIZON_TIME
        else:
            action_limit = len(request.actions)
            requested_step_limit = action_limit if request.horizon_steps is None else min(action_limit, int(request.horizon_steps))
            effective_limit = min(requested_step_limit, self.max_rollout_depth)
            engine_limited = effective_limit < requested_step_limit
            if engine_limited:
                warnings.append(
                    f"configured max_rollout_depth={self.max_rollout_depth} limited execution"
                )

            for action_index in range(effective_limit):
                check_timeout(started_monotonic, timeout, operation="simulation_run")
                if self._call_run_predicate(request.cancellation_check, name="cancellation_check", state=state, step=action_index, simulation_time=simulation_time):
                    termination = TerminationReason.CANCELLED
                    break
                if request.horizon_time is not None and simulation_time + request.time_step > float(request.horizon_time) + 1.0e-12:
                    termination = TerminationReason.HORIZON_TIME
                    break

                active = self._active_interventions(request, action_index, simulation_time)
                outcome = self.transition.apply(
                    model,
                    state=state,
                    action=request.actions[action_index],
                    interventions=active,
                    parameters=request.parameters,
                    rng=generator,
                    step=action_index,
                    simulation_time=simulation_time,
                    dt=request.time_step,
                )
                state = outcome.state
                simulation_time += request.time_step
                steps.append(
                    TrajectoryStep(
                        index=action_index + 1,
                        simulation_time=simulation_time,
                        state=clone_state(state),
                        action=clone_state(request.actions[action_index]),
                        interventions=active,
                        metadata=dict(outcome.metadata),
                    )
                )

                if outcome.terminal:
                    termination = TerminationReason.MODEL_TERMINATED
                    break
                if self._call_run_predicate(request.termination_condition, name="termination_condition", state=state, step=action_index + 1, simulation_time=simulation_time):
                    termination = TerminationReason.USER_CONDITION
                    break
                if request.horizon_time is not None and simulation_time >= float(request.horizon_time) - 1.0e-12:
                    termination = TerminationReason.HORIZON_TIME
                    break
                if request.horizon_steps is not None and action_index + 1 >= int(request.horizon_steps):
                    termination = TerminationReason.HORIZON_STEPS
                    break
            else:
                if engine_limited:
                    termination = TerminationReason.ENGINE_LIMIT
                elif request.horizon_steps is not None and effective_limit >= int(request.horizon_steps) and int(request.horizon_steps) < action_limit:
                    termination = TerminationReason.HORIZON_STEPS
                else:
                    termination = TerminationReason.ACTIONS_EXHAUSTED

        finished_at = utc_now_iso()
        duration = time.monotonic() - started_monotonic
        final_rng_state = rng_state_snapshot(generator)
        trajectory = Trajectory(
            trajectory_id=f"trajectory-{uuid.uuid4().hex[:20]}",
            steps=tuple(steps),
            termination_reason=termination,
            seed=concrete_seed,
            model_id=model_id,
            scenario_id=request.scenario_id,
            warnings=tuple(warnings),
        )
        runtime = SimulationRuntimeMetadata(
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=float(duration),
            transition_count=trajectory.transition_count,
            deterministic=request.deterministic,
            numpy_version=np.__version__,
            python_version=sys.version.split()[0],
            rng_bit_generator=type(generator.bit_generator).__name__,
        )
        model_snapshot: Mapping[str, Any] = {}
        snapshot = getattr(model, "snapshot", None)
        if callable(snapshot):
            try:
                candidate = snapshot()
                if isinstance(candidate, Mapping):
                    model_snapshot = dict(candidate)
            except Exception as exc:
                warnings.append(f"model snapshot unavailable: {type(exc).__name__}")
                logger.warning("Model snapshot failed | run_id=%s error=%s", request.run_id, exc)
        result = SimulationResult(
            run_id=request.run_id,
            scenario_id=request.scenario_id,
            model_id=model_id,
            model_version=model_version,
            trajectory=trajectory,
            actions_supplied=tuple(clone_state(value) for value in request.actions),
            interventions_supplied=tuple(request.interventions),
            parameters=clone_state(dict(request.parameters)),
            seed=concrete_seed,
            rng_state_initial=initial_rng_state,
            rng_state_final=final_rng_state,
            runtime=runtime,
            warnings=tuple(warnings),
            branch_metadata=dict(branch_metadata or {}),
            metadata={
                "request_metadata": dict(request.metadata),
                "model_snapshot": model_snapshot,
                **dict(metadata or {}),
            },
        )
        logger.info(
            "Simulation end | run_id=%s transitions=%s termination=%s duration=%.6fs",
            request.run_id,
            trajectory.transition_count,
            termination.value,
            duration,
        )
        if self.pretty_progress:
            printer.status("DONE", f"Simulation {request.run_id} ended: {termination.value}", "success")
        return result


__all__ = ["BaseEngine"]
