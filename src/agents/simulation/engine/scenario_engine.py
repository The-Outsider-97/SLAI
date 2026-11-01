"""Bounded scenario construction and branching execution.

Scenario branching is descriptive simulation, not planning.  Perturbation and
screening concepts are informed by Morris (1991), Saltelli et al. (2008), and
McKay et al. (1979).  Parameter randomization is compatible with the sim-to-real
methodology of Tobin et al. (2017) and Peng et al. (2018).  Branch limits prevent
uncontrolled combinatorial growth; all parameter values and interventions are
externally supplied or sampled from explicitly supplied experiment bounds.
"""

from __future__ import annotations

__version__ = "2.3.0"

import uuid

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, Optional

from .base_engine import BaseEngine
from ..modules.monte_carlo import MonteCarlo
from ..simulation_types import *
from ..utils.config_loader import get_config_section, load_global_config
from ..utils.simulation_errors import *
from ..utils.simulation_helpers import *
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Scenario Engine")


class ScenarioEngine(BaseEngine):
    """Execute bounded scenario forests over one supplied simulation model."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        global_config = load_global_config()
        section = dict(get_config_section("scenario_engine", config=global_config) or {})
        if config:
            section.update(dict(config))
        base_config = dict(section.get("base_engine", {}) or {})
        super().__init__(base_config)
        self.max_branches = int(section.get("max_branches", 256))
        self.max_depth = int(section.get("max_depth", 8))
        if self.max_branches <= 0 or self.max_depth < 0:
            raise SimulationValidationError("scenario max_branches must be > 0 and max_depth >= 0")
        self.monte_carlo = MonteCarlo(dict(section.get("sampling", {}) or {}))

    @staticmethod
    def _scenario_parameters(scenario: Scenario) -> dict[str, Any]:
        parameters = dict(scenario.parameters)
        for perturbation in scenario.perturbations:
            parameters[perturbation.parameter] = perturbation.value
        return parameters

    def run_scenario(
        self,
        request: SimulationRequest,
        model: Any,
        scenario: Scenario,
        *,
        depth: int = 0,
    ) -> SimulationResult:
        if not isinstance(scenario, Scenario):
            raise SimulationValidationError("scenario must be Scenario")
        if depth < 0 or depth > self.max_depth:
            raise SimulationBranchLimitError(
                "scenario depth exceeds configured maximum",
                context={"scenario_id": scenario.scenario_id, "depth": depth, "max_depth": self.max_depth},
            )
        scenario_seed = normalize_seed(request.seed if scenario.seed is None else scenario.seed)
        scenario_request = replace(
            request,
            run_id=f"{request.run_id}-scenario-{scenario.scenario_id}-{uuid.uuid4().hex[:8]}",
            scenario_id=scenario.scenario_id,
            seed=scenario_seed,
            parameters=merge_parameters(request.parameters, self._scenario_parameters(scenario)),
            interventions=tuple(request.interventions) + tuple(scenario.interventions),
            metadata={**dict(request.metadata), "scenario_metadata": dict(scenario.metadata)},
        )
        logger.info(
            "Scenario execution | scenario=%s parent=%s depth=%s seed=%s",
            scenario.scenario_id,
            scenario.parent_scenario_id,
            depth,
            scenario_seed,
        )
        return self.run(
            scenario_request,
            model,
            seed=scenario_seed,
            branch_metadata={
                "scenario_id": scenario.scenario_id,
                "parent_scenario_id": scenario.parent_scenario_id,
                "depth": depth,
                "perturbations": [item.to_dict() for item in scenario.perturbations],
            },
        )

    def _validate_forest(self, scenarios: Sequence[Scenario]) -> tuple[dict[str, Scenario], dict[str, int], dict[str, tuple[str, ...]]]:
        if not scenarios:
            raise SimulationValidationError("at least one scenario is required")
        if len(scenarios) > self.max_branches:
            raise SimulationBranchLimitError(
                "scenario count exceeds configured maximum",
                context={"count": len(scenarios), "max_branches": self.max_branches},
            )
        by_id: dict[str, Scenario] = {}
        children: dict[str, list[str]] = defaultdict(list)
        for scenario in scenarios:
            if not isinstance(scenario, Scenario):
                raise SimulationValidationError("all scenarios must be Scenario instances")
            if scenario.scenario_id in by_id:
                raise SimulationValidationError("scenario identifiers must be unique", context={"scenario_id": scenario.scenario_id})
            by_id[scenario.scenario_id] = scenario
        for scenario in scenarios:
            parent = scenario.parent_scenario_id
            if parent is not None:
                if parent not in by_id:
                    raise SimulationValidationError(
                        "scenario parent is absent from the submitted forest",
                        context={"scenario_id": scenario.scenario_id, "parent_scenario_id": parent},
                    )
                children[parent].append(scenario.scenario_id)

        depths: dict[str, int] = {}
        for scenario_id in by_id:
            chain: set[str] = set()
            current = scenario_id
            depth = 0
            while by_id[current].parent_scenario_id is not None:
                if current in chain:
                    raise SimulationValidationError("cycle detected in scenario lineage", context={"scenario_id": scenario_id})
                chain.add(current)
                parent = by_id[current].parent_scenario_id
                assert parent is not None
                current = parent
                depth += 1
                if depth > self.max_depth:
                    raise SimulationBranchLimitError(
                        "scenario lineage exceeds configured maximum depth",
                        context={"scenario_id": scenario_id, "depth": depth, "max_depth": self.max_depth},
                    )
            depths[scenario_id] = depth
        child_map = {key: tuple(sorted(value)) for key, value in children.items()}
        return by_id, depths, child_map

    def execute_scenarios(
        self,
        request: SimulationRequest,
        model: Any,
        scenarios: Sequence[Scenario],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ScenarioBatchResult:
        by_id, depths, children = self._validate_forest(tuple(scenarios))
        results: dict[str, SimulationResult] = {}
        branches: dict[str, ScenarioBranch] = {}
        ordered_ids = sorted(by_id, key=lambda item: (depths[item], item))
        for scenario_id in ordered_ids:
            scenario = by_id[scenario_id]
            result = self.run_scenario(request, model, scenario, depth=depths[scenario_id])
            results[scenario_id] = result
            branches[scenario_id] = ScenarioBranch(
                scenario_id=scenario_id,
                parent_scenario_id=scenario.parent_scenario_id,
                depth=depths[scenario_id],
                child_ids=children.get(scenario_id, ()),
                metadata=scenario.metadata,
            )
        return ScenarioBatchResult(
            batch_id=f"scenario-batch-{uuid.uuid4().hex[:20]}",
            results=results,
            branches=branches,
            metadata={"scenario_count": len(results), **dict(metadata or {})},
        )

    def oat_scenarios(
        self,
        baseline: Scenario,
        parameter_values: Mapping[str, Sequence[Any]],
    ) -> tuple[Scenario, ...]:
        """Create one-at-a-time branches from explicitly supplied values."""
        scenarios: list[Scenario] = [baseline]
        for parameter, values in parameter_values.items():
            for index, value in enumerate(values):
                scenarios.append(
                    Scenario(
                        scenario_id=f"{baseline.scenario_id}-oat-{parameter}-{index:04d}",
                        parent_scenario_id=baseline.scenario_id,
                        perturbations=(Perturbation(parameter=str(parameter), value=value),),
                        metadata={"experiment": "one_at_a_time", "parameter": str(parameter)},
                    )
                )
        if len(scenarios) > self.max_branches:
            raise SimulationBranchLimitError(
                "OAT design exceeds scenario branch limit",
                context={"count": len(scenarios), "max_branches": self.max_branches},
            )
        return tuple(scenarios)

    def latin_hypercube_scenarios(
        self,
        baseline: Scenario,
        bounds: Mapping[str, tuple[float, float]],
        *,
        sample_count: int,
        seed: Optional[int] = None,
    ) -> tuple[Scenario, ...]:
        """Create bounded LHS parameter scenarios without objective optimization."""
        if sample_count + 1 > self.max_branches:
            raise SimulationBranchLimitError(
                "Latin hypercube design exceeds scenario branch limit",
                context={"sample_count": sample_count, "max_branches": self.max_branches},
            )
        rows, root_seed = self.monte_carlo.latin_hypercube(bounds, sample_count, seed=seed)
        _, child_seeds = spawn_child_seeds(root_seed, sample_count)
        scenarios: list[Scenario] = [baseline]
        for index, row in enumerate(rows):
            scenarios.append(
                Scenario(
                    scenario_id=f"{baseline.scenario_id}-lhs-{index:04d}",
                    parent_scenario_id=baseline.scenario_id,
                    parameters=row,
                    seed=child_seeds[index],
                    metadata={"experiment": "latin_hypercube", "sample_index": index, "root_seed": root_seed},
                )
            )
        return tuple(scenarios)


__all__ = ["ScenarioEngine"]
