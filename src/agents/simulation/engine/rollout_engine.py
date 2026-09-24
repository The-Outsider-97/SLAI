"""Trajectory rollout engine for deterministic, stochastic, and counterfactual runs.

Trajectory semantics follow Puterman (1994).  Learned-model imagination is
limited to state rollout concepts from Ha & Schmidhuber (2018) and Hafner et
al. (2020); no control-policy training or action selection is implemented.
Counterfactual execution accepts externally defined interventions only; Pearl's
intervention semantics are a boundary reference, not a causal-discovery engine.
"""

from __future__ import annotations

__version__ = "2.3.0"

import uuid

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Optional

from .base_engine import BaseEngine
from ..modules.monte_carlo import MonteCarlo
from ..simulation_types import *
from ..utils.config_loader import get_config_section, load_global_config
from ..utils.simulation_errors import SimulationValidationError
from ..utils.simulation_helpers import *
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Rollout Engine")


@dataclass(frozen=True, slots=True)
class CounterfactualRollout:
    """Baseline and externally intervened trajectories; no causal attribution."""

    baseline: Optional[SimulationResult]
    alternate: SimulationResult
    interventions: tuple[Intervention, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": None if self.baseline is None else self.baseline.to_dict(),
            "alternate": self.alternate.to_dict(),
            "interventions": [item.to_dict() for item in self.interventions],
            "metadata": to_json_safe(dict(self.metadata)),
        }


class RolloutEngine(BaseEngine):
    """High-level rollout operations over :class:`BaseEngine`."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        global_config = load_global_config()
        section = dict(get_config_section("rollout_engine", config=global_config) or {})
        if config:
            section.update(dict(config))
        base_config = dict(section.get("base_engine", {}) or {})
        super().__init__(base_config)
        monte_carlo_config = dict(section.get("monte_carlo", {}) or {})
        self.monte_carlo = MonteCarlo(monte_carlo_config)

    def rollout(self, request: SimulationRequest, model: Any, **kwargs: Any) -> SimulationResult:
        logger.debug("Rollout start | run_id=%s", request.run_id)
        return self.run(request, model, **kwargs)

    def rollout_many(
        self,
        request: SimulationRequest,
        model: Any,
        *,
        sample_count: Optional[int] = None,
        seed: Optional[int] = None,
        concurrency: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        continue_on_error: Optional[bool] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> MonteCarloResult:
        """Run the same externally supplied action sequence over independent streams."""
        root_seed = request.seed if seed is None else seed
        effective_concurrency = self.monte_carlo.default_concurrency if concurrency is None else int(concurrency)
        if effective_concurrency > 1 and not bool(getattr(model, "thread_safe", False)):
            raise SimulationValidationError(
                "concurrent rollouts require model.thread_safe=True; use concurrency=1 for mutable environments/models",
                context={"model_type": type(model).__name__, "concurrency": effective_concurrency},
            )

        def one_rollout(*, sample_index: int, seed: int) -> SimulationResult:
            child_request = replace(
                request,
                run_id=f"{request.run_id}-sample-{sample_index:06d}",
                seed=seed,
            )
            return self.run(
                child_request,
                model,
                seed=seed,
                metadata={"monte_carlo_sample_index": sample_index},
            )

        return self.monte_carlo.run(
            one_rollout,
            sample_count=sample_count,
            seed=root_seed,
            concurrency=effective_concurrency,
            timeout_seconds=timeout_seconds,
            continue_on_error=continue_on_error,
            cancellation_check=request.cancellation_check,
            metadata=metadata,
        )

    def counterfactual_rollout(
        self,
        request: SimulationRequest,
        model: Any,
        interventions: Sequence[Intervention],
        *,
        run_baseline: bool = True,
        alternate_scenario_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> CounterfactualRollout:
        """Execute an externally specified intervention against a common RNG seed."""
        supplied = tuple(interventions)
        if not supplied:
            raise SimulationValidationError("counterfactual rollout requires at least one external intervention")
        if any(not isinstance(item, Intervention) for item in supplied):
            raise SimulationValidationError("counterfactual interventions must be Intervention instances")
        common_seed = normalize_seed(request.seed)
        baseline_request = replace(request, seed=common_seed)
        baseline = self.run(baseline_request, model, seed=common_seed) if run_baseline else None
        alternate = replace(
            request,
            run_id=f"{request.run_id}-counterfactual-{uuid.uuid4().hex[:10]}",
            scenario_id=alternate_scenario_id or request.scenario_id,
            seed=common_seed,
            interventions=tuple(request.interventions) + supplied,
        )
        alternate_result = self.run(
            alternate,
            model,
            seed=common_seed,
            metadata={"counterfactual": True, **dict(metadata or {})},
        )
        return CounterfactualRollout(
            baseline=baseline,
            alternate=alternate_result,
            interventions=supplied,
            metadata={"shared_seed": common_seed, **dict(metadata or {})},
        )

    def perturbed_rollout(
        self,
        request: SimulationRequest,
        model: Any,
        parameter_overrides: Mapping[str, Any],
        *,
        run_id: Optional[str] = None,
        scenario_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> SimulationResult:
        """Execute an explicitly supplied parameter perturbation without optimization."""
        if not isinstance(parameter_overrides, Mapping):
            raise SimulationValidationError("parameter_overrides must be a mapping")
        concrete_seed = normalize_seed(request.seed if seed is None else seed)
        perturbed = replace(
            request,
            run_id=run_id or f"{request.run_id}-perturbed-{uuid.uuid4().hex[:10]}",
            scenario_id=request.scenario_id if scenario_id is None else scenario_id,
            seed=concrete_seed,
            parameters=merge_parameters(request.parameters, parameter_overrides),
        )
        return self.run(
            perturbed,
            model,
            seed=concrete_seed,
            metadata={"parameter_perturbation": to_json_safe(dict(parameter_overrides))},
        )

    def replay(self, result: SimulationResult, model: Any) -> SimulationResult:
        """Replay a recorded result's supplied inputs using its recorded integer seed."""
        if not isinstance(result, SimulationResult):
            raise SimulationValidationError("result must be SimulationResult")
        request = SimulationRequest(
            initial_state=result.initial_state,
            actions=result.actions_supplied,
            parameters=result.parameters,
            interventions=result.interventions_supplied,
            run_id=f"{result.run_id}-replay-{uuid.uuid4().hex[:10]}",
            scenario_id=result.scenario_id,
            seed=result.seed,
            mode=SimulationMode.DETERMINISTIC if result.runtime.deterministic else SimulationMode.STOCHASTIC,
            time_step=(
                result.trajectory.steps[1].simulation_time - result.trajectory.steps[0].simulation_time
                if len(result.trajectory.steps) > 1
                else 1.0
            ),
            horizon_steps=result.trajectory.transition_count,
            metadata={"replay_of": result.run_id},
        )
        return self.run(request, model, seed=result.seed, metadata={"replay_of": result.run_id})


__all__ = ["CounterfactualRollout", "RolloutEngine"]
