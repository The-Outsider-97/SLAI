"""
Production orchestration facade for the SLAI v2.3 Simulation subsystem.

Architectural boundary
----------------------
SimulationAgent accepts externally supplied simulation intent and delegates
state propagation, Monte Carlo execution, scenario branching, perturbation,
counterfactual rollout, sensitivity analysis, and replay to the Simulation
subsystem.  It does not generate or rank plans, optimize objective functions,
perform causal discovery, evaluate desirability, formally verify properties,
gate safety, perceive the physical world, estimate state, or own spatial maps.

Configuration boundary
----------------------
Agent-level configuration is read only from ``agents_config.yaml`` through
BaseAgent's main configuration loader.  This module deliberately does not import
or reference the Simulation subsystem configuration facade or any subsystem-owned
configuration file. Subsystem components remain responsible for their own internal
configuration.

Memory boundary
---------------
``self.shared_memory`` is BaseAgent's cross-agent communication fabric.
``self.sim_memory`` is Simulation-specific bounded lineage/replay metadata.
Completed full SimulationResult objects are returned to callers; SharedMemory
receives only bounded summaries/references unless bounded trajectory publication
is explicitly enabled.

The architecture follows the model/simulator separation of Zeigler, Praehofer
& Kim (2000), simulation run discipline from Law (2015), downstream
verification/validation separation from Sargent (2013), and reproducibility
principles from Sandve et al. (2013).
"""

from __future__ import annotations

__version__ = "2.3.0"

import math
import uuid

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TypeAlias, cast

from .base_agent import BaseAgent
from .base.utils.base_errors import *
from .base.utils.base_helpers import *
from .base.utils.main_config_loader import get_config_section
from .simulation import *
from .simulation.utils.simulation_errors import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("Simulation Agent")
printer = PrettyPrinter()


class SimulationOperation(str, Enum):
    """Operations exposed by the orchestration facade."""

    ROLLOUT = "rollout"
    TRANSITION = "transition"
    MONTE_CARLO = "monte_carlo"
    SCENARIO = "scenario"
    COUNTERFACTUAL = "counterfactual"
    PERTURBATION = "perturbation"
    SENSITIVITY = "sensitivity"
    ANALYSIS = "analysis"
    REPLAY = "replay"


SimulationOutcome: TypeAlias = (
    SimulationResult
    | MonteCarloResult
    | ScenarioBatchResult
    | CounterfactualRollout
    | SensitivityEstimate
    | Mapping[str, Any]
)


@dataclass(frozen=True, slots=True)
class SimulationAgentRequest:
    """Normalized Agent-facing Simulation request.

    Numerical details remain in subsystem-native objects.  The facade adds only
    orchestration identity and routing information.
    """

    operation: SimulationOperation
    simulation: Optional[SimulationRequest] = None
    model: Optional[SimulationModel] = None
    model_id: Optional[str] = None
    scenario: Optional[Scenario] = None
    scenarios: tuple[Scenario, ...] = ()
    interventions: tuple[Intervention, ...] = ()
    parameter_overrides: Mapping[str, Any] = field(default_factory=dict)
    source_result: Optional[SimulationResult] = None
    options: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: f"simreq-{uuid.uuid4().hex[:20]}")

    def __post_init__(self) -> None:
        operation = self.operation
        if not isinstance(operation, SimulationOperation):
            try:
                operation = SimulationOperation(str(operation).strip().lower())
            except (TypeError, ValueError) as exc:
                raise BaseValidationError(
                    "Unsupported SimulationAgent operation.",
                    component="SimulationAgent",
                    context={"operation": str(self.operation)},
                    cause=exc,
                ) from exc
            object.__setattr__(self, "operation", operation)

        request_id = str(self.request_id or "").strip()
        if not request_id:
            raise BaseValidationError("SimulationAgent request_id must be non-empty.", component="SimulationAgent")
        object.__setattr__(self, "request_id", request_id)

        model_id = None if self.model_id is None else str(self.model_id).strip()
        if model_id == "":
            model_id = None
        object.__setattr__(self, "model_id", model_id)

        if self.simulation is not None and not isinstance(self.simulation, SimulationRequest):
            raise BaseValidationError("simulation must be SimulationRequest or null.", component="SimulationAgent")
        if self.source_result is not None and not isinstance(self.source_result, SimulationResult):
            raise BaseValidationError("source_result must be SimulationResult or null.", component="SimulationAgent")
        if self.scenario is not None and not isinstance(self.scenario, Scenario):
            raise BaseValidationError("scenario must be Scenario or null.", component="SimulationAgent")
        if any(not isinstance(item, Scenario) for item in self.scenarios):
            raise BaseValidationError("scenarios must contain Scenario objects.", component="SimulationAgent")
        if any(not isinstance(item, Intervention) for item in self.interventions):
            raise BaseValidationError("interventions must contain Intervention objects.", component="SimulationAgent")
        if not isinstance(self.parameter_overrides, Mapping):
            raise BaseValidationError("parameter_overrides must be a mapping.", component="SimulationAgent")
        if not isinstance(self.options, Mapping) or not isinstance(self.metadata, Mapping):
            raise BaseValidationError("options and metadata must be mappings.", component="SimulationAgent")

        object.__setattr__(self, "scenarios", tuple(self.scenarios))
        object.__setattr__(self, "interventions", tuple(self.interventions))
        object.__setattr__(self, "parameter_overrides", dict(self.parameter_overrides))
        object.__setattr__(self, "options", dict(self.options))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SimulationAgentRequest":
        if not isinstance(payload, Mapping):
            raise BaseValidationError("SimulationAgent request must be a mapping.", component="SimulationAgent")

        raw_operation = payload.get("operation", "rollout")
        try:
            operation = (
                raw_operation
                if isinstance(raw_operation, SimulationOperation)
                else SimulationOperation(str(raw_operation).strip().lower())
            )
        except (TypeError, ValueError) as exc:
            raise BaseValidationError(
                "Unsupported SimulationAgent operation.",
                component="SimulationAgent",
                context={"operation": str(raw_operation)},
                cause=exc,
            ) from exc

        raw_simulation = payload.get("simulation", payload.get("request"))
        simulation: Optional[SimulationRequest]
        if isinstance(raw_simulation, SimulationRequest):
            simulation = raw_simulation
        elif isinstance(raw_simulation, Mapping):
            simulation = cls._simulation_request_from_mapping(raw_simulation)
        elif raw_simulation is None and cls._looks_like_simulation_request(payload):
            simulation_payload = dict(payload)
            if operation is SimulationOperation.COUNTERFACTUAL:
                # Compact counterfactual requests use top-level interventions for
                # the alternate world only. Baseline interventions, if any, must
                # be explicit inside a nested ``simulation`` request.
                simulation_payload.pop("interventions", None)
            simulation = cls._simulation_request_from_mapping(simulation_payload)
        elif raw_simulation is None:
            simulation = None
        else:
            raise BaseValidationError(
                "simulation/request must be SimulationRequest, a mapping, or null.",
                component="SimulationAgent",
            )

        scenario = cls._scenario_from_value(payload.get("scenario"))
        scenarios = cls._scenario_sequence(payload.get("scenarios", ()))
        interventions = cls._intervention_sequence(payload.get("interventions", ()))

        options = payload.get("options", {})
        metadata = payload.get("metadata", {})
        parameter_overrides = payload.get("parameter_overrides", {})
        if not isinstance(options, Mapping):
            raise BaseValidationError("options must be a mapping.", component="SimulationAgent")
        if not isinstance(metadata, Mapping):
            raise BaseValidationError("metadata must be a mapping.", component="SimulationAgent")
        if not isinstance(parameter_overrides, Mapping):
            raise BaseValidationError("parameter_overrides must be a mapping.", component="SimulationAgent")

        return cls(
            operation=operation,
            simulation=simulation,
            model=payload.get("model"),
            model_id=payload.get("model_id"),
            scenario=scenario,
            scenarios=scenarios,
            interventions=interventions,
            parameter_overrides=dict(parameter_overrides),
            source_result=payload.get("source_result"),
            options=dict(options),
            metadata=dict(metadata),
            request_id=str(payload.get("request_id") or f"simreq-{uuid.uuid4().hex[:20]}"),
        )

    @staticmethod
    def _looks_like_simulation_request(payload: Mapping[str, Any]) -> bool:
        return "initial_state" in payload and "actions" in payload

    @classmethod
    def _simulation_request_from_mapping(cls, payload: Mapping[str, Any]) -> SimulationRequest:
        if "initial_state" not in payload:
            raise BaseValidationError("Simulation request requires initial_state.", component="SimulationAgent")
        if "actions" not in payload:
            raise BaseValidationError("Simulation request requires actions.", component="SimulationAgent")

        raw_mode = payload.get("mode", SimulationMode.DETERMINISTIC)
        try:
            mode = raw_mode if isinstance(raw_mode, SimulationMode) else SimulationMode(str(raw_mode).strip().lower())
        except (TypeError, ValueError) as exc:
            raise BaseValidationError("Simulation request mode must be deterministic or stochastic.", component="SimulationAgent", cause=exc) from exc

        return SimulationRequest(
            initial_state=payload["initial_state"],
            actions=payload["actions"],
            parameters=dict(payload.get("parameters", {}) or {}),
            interventions=cls._intervention_sequence(payload.get("interventions", ())),
            run_id=str(payload.get("run_id") or f"sim-{uuid.uuid4().hex[:20]}"),
            scenario_id=payload.get("scenario_id"),
            seed=payload.get("seed"),
            mode=mode,
            time_step=float(payload.get("time_step", 1.0)),
            horizon_steps=payload.get("horizon_steps"),
            horizon_time=payload.get("horizon_time"),
            timeout_seconds=payload.get("timeout_seconds"),
            termination_condition=payload.get("termination_condition"),
            cancellation_check=payload.get("cancellation_check"),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    @staticmethod
    def _intervention_from_value(value: Any) -> Optional[Intervention]:
        if value is None:
            return None
        if isinstance(value, Intervention):
            return value
        if isinstance(value, Mapping):
            if "payload" not in value:
                raise BaseValidationError("Intervention mapping requires payload.", component="SimulationAgent")
            return Intervention(
                payload=value["payload"],
                intervention_id=str(value.get("intervention_id") or f"intervention-{uuid.uuid4().hex[:16]}"),
                step=value.get("step"),
                time=value.get("time"),
                metadata=dict(value.get("metadata", {}) or {}),
            )
        raise BaseValidationError("Intervention must be an Intervention or mapping.", component="SimulationAgent")

    @classmethod
    def _intervention_sequence(cls, values: Any) -> tuple[Intervention, ...]:
        if values is None:
            return ()
        if isinstance(values, (Intervention, Mapping)):
            item = cls._intervention_from_value(values)
            return () if item is None else (item,)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
            normalized: list[Intervention] = []
            for value in values:
                item = cls._intervention_from_value(value)
                if item is not None:
                    normalized.append(item)
            return tuple(normalized)
        raise BaseValidationError("interventions must be a sequence.", component="SimulationAgent")

    @classmethod
    def _scenario_from_value(cls, value: Any) -> Optional[Scenario]:
        if value is None:
            return None
        if isinstance(value, Scenario):
            return value
        if not isinstance(value, Mapping):
            raise BaseValidationError("Scenario must be a Scenario or mapping.", component="SimulationAgent")
        scenario_id = str(value.get("scenario_id") or "").strip()
        if not scenario_id:
            raise BaseValidationError("Scenario mapping requires scenario_id.", component="SimulationAgent")

        raw_perturbations = value.get("perturbations", ())
        perturbations: list[Perturbation] = []
        if isinstance(raw_perturbations, Sequence) and not isinstance(
            raw_perturbations, (str, bytes, bytearray)
        ):
            for item in raw_perturbations:
                if isinstance(item, Perturbation):
                    perturbations.append(item)
                elif isinstance(item, Mapping):
                    perturbations.append(
                        Perturbation(
                            parameter=str(item.get("parameter") or ""),
                            value=item.get("value"),
                            perturbation_id=str(item.get("perturbation_id") or f"perturbation-{uuid.uuid4().hex[:16]}"),
                            metadata=dict(item.get("metadata", {}) or {}),
                        )
                    )
                else:
                    raise BaseValidationError("Scenario perturbations must be Perturbation objects or mappings.", component="SimulationAgent")
        else:
            raise BaseValidationError("Scenario perturbations must be a sequence.", component="SimulationAgent")

        return Scenario(
            scenario_id=scenario_id,
            parent_scenario_id=value.get("parent_scenario_id"),
            parameters=dict(value.get("parameters", {}) or {}),
            interventions=cls._intervention_sequence(value.get("interventions", ())),
            perturbations=tuple(perturbations),
            seed=value.get("seed"),
            metadata=dict(value.get("metadata", {}) or {}),
        )

    @classmethod
    def _scenario_sequence(cls, values: Any) -> tuple[Scenario, ...]:
        if values is None:
            return ()
        if isinstance(values, (Scenario, Mapping)):
            scenario = cls._scenario_from_value(values)
            return () if scenario is None else (scenario,)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
            scenarios: list[Scenario] = []
            for value in values:
                scenario = cls._scenario_from_value(value)
                if scenario is not None:
                    scenarios.append(scenario)
            return tuple(scenarios)
        raise BaseValidationError("scenarios must be a sequence.", component="SimulationAgent")


class SimulationAgent(BaseAgent):
    """Production orchestration boundary over ``src.agents.simulation``."""

    # The current SimulationMemory exposes bounded runtime records but no public
    # restore/import contract for reconstructing typed SimulationResult objects.
    # Declaring checkpoint support here would therefore overstate restore
    # semantics.  Durable persistence remains owned by BaseAgent/checkpointing.
    CHECKPOINTING_SUPPORTED = False

    def __init__(
        self,
        shared_memory: Any,
        agent_factory: Any,
        config: Optional[Mapping[str, Any]] = None,
        *,
        vocabulary: type[SimulationVocab] | SimulationVocab = SimulationVocab,
        checkpoint_manager: Any = None,
    ) -> None:
        # BaseAgent owns only the base_agent section.  Agent-specific overrides
        # are intentionally not passed into BaseAgent.
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            checkpoint_manager=checkpoint_manager,
        )

        self.agent_config: dict[str, Any] = dict(get_config_section("simulation_agent") or {})
        if config is not None:
            if not isinstance(config, Mapping):
                raise BaseConfigurationError("SimulationAgent config override must be a mapping.", component=self.name)
            self.agent_config.update(dict(config))

        self._load_agent_config()
        self._validate_agent_config()

        self.vocab = vocabulary() if isinstance(vocabulary, type) else vocabulary

        # Simulation subsystem services.  They own their own internal
        # subsystem-owned configuration; this Agent does not read it.
        self.sim_memory = SimulationMemory(
            {
                "max_records": self.max_recent_runs,
                "store_request_payloads": self.store_request_payloads,
                "store_result_payloads": self.retain_full_results_in_sim_memory,
            }
        )
        self.model_registry = SimulationModelRegistry()
        self.rollout_engine = RolloutEngine()
        self.scenario_engine = ScenarioEngine()
        self.analysis = SimulationAnalysis()

        self._last_operation: Optional[str] = None
        self._last_run_id: Optional[str] = None
        self._last_failure: Optional[dict[str, Any]] = None
        self._completed_operations = 0

        self._dispatch: dict[
            SimulationOperation,
            Callable[[SimulationAgentRequest], SimulationOutcome],
        ] = {
            SimulationOperation.ROLLOUT: self._dispatch_rollout,
            SimulationOperation.TRANSITION: self._dispatch_transition,
            SimulationOperation.MONTE_CARLO: self._dispatch_monte_carlo,
            SimulationOperation.SCENARIO: self._dispatch_scenario,
            SimulationOperation.COUNTERFACTUAL: self._dispatch_counterfactual,
            SimulationOperation.PERTURBATION: self._dispatch_perturbation,
            SimulationOperation.SENSITIVITY: self._dispatch_sensitivity,
            SimulationOperation.ANALYSIS: self._dispatch_analysis,
            SimulationOperation.REPLAY: self._dispatch_replay,
        }

        logger.info(
            "Simulation Agent initialized | publish_results=%s | sim_memory_capacity=%s",
            self.publish_results,
            self.max_recent_runs,
        )
        self._publish_event(
            "initialized",
            {
                "agent_id": self.agent_id,
                "checkpointing_supported": self.supports_checkpointing,
            },
        )

    # ------------------------------------------------------------------
    # Agent-level configuration: agents_config.yaml only
    # ------------------------------------------------------------------

    def _cfg(self, key: str, default: Any) -> Any:
        return self.agent_config.get(key, default)

    def _load_agent_config(self) -> None:
        self.enabled = coerce_bool(self._cfg("enabled", True), True)
        self.publish_lifecycle_events = coerce_bool(
            self._cfg("publish_lifecycle_events", True), True
        )
        self.publish_results = coerce_bool(self._cfg("publish_results", True), True)
        self.publish_errors = coerce_bool(self._cfg("publish_errors", True), True)

        self.shared_memory_ttl_seconds = coerce_int(
            self._cfg("shared_memory_ttl_seconds", 86400),
            86400,
            minimum=0,
            maximum=31_536_000,
        )
        self.max_recent_runs = coerce_int(
            self._cfg("max_recent_runs", 256),
            256,
            minimum=1,
            maximum=100_000,
        )
        self.store_request_payloads = coerce_bool(
            self._cfg("store_request_payloads", True), True
        )
        self.retain_full_results_in_sim_memory = coerce_bool(
            self._cfg("retain_full_results_in_sim_memory", False), False
        )

        self.include_trajectory_in_shared_memory = coerce_bool(
            self._cfg("include_trajectory_in_shared_memory", False), False
        )
        self.max_shared_trajectory_steps = coerce_int(
            self._cfg("max_shared_trajectory_steps", 64),
            64,
            minimum=1,
            maximum=10_000,
        )
        self.max_shared_list_items = coerce_int(
            self._cfg("max_shared_list_items", 128),
            128,
            minimum=1,
            maximum=100_000,
        )
        self.max_shared_string_length = coerce_int(
            self._cfg("max_shared_string_length", 4096),
            4096,
            minimum=128,
            maximum=1_000_000,
        )

        default_operation = str(self._cfg("default_operation", "rollout")).strip().lower()
        try:
            self.default_operation = SimulationOperation(default_operation)
        except ValueError as exc:
            raise BaseConfigurationError(
                "simulation_agent.default_operation is unsupported.",
                component=self.name,
                context={"value": default_operation},
                cause=exc,
            ) from exc

        self.event_channel = str(
            self._cfg("event_channel", "simulation.events")
            or "simulation.events"
        ).strip()
        self.result_key_prefix = str(
            self._cfg("result_key_prefix", "simulation_agent.result")
            or "simulation_agent.result"
        ).strip()
        self.summary_key_prefix = str(
            self._cfg("summary_key_prefix", "simulation_agent.summary")
            or "simulation_agent.summary"
        ).strip()
        self.error_key_prefix = str(
            self._cfg("error_key_prefix", "simulation_agent.error")
            or "simulation_agent.error"
        ).strip()
        self.latest_result_key = str(
            self._cfg("latest_result_key", "simulation_agent.latest")
            or "simulation_agent.latest"
        ).strip()

    def _validate_agent_config(self) -> None:
        for field_name in (
            "event_channel",
            "result_key_prefix",
            "summary_key_prefix",
            "error_key_prefix",
            "latest_result_key",
        ):
            if not getattr(self, field_name):
                raise BaseConfigurationError(
                    f"simulation_agent.{field_name} must be non-empty.",
                    component=self.name,
                    context={"field": field_name},
                )

        if (self.publish_results or self.publish_errors or self.publish_lifecycle_events):
            missing = [
                name
                for name in ("set", "publish")
                if not callable(getattr(self.shared_memory, name, None))
            ]
            if missing:
                raise BaseConfigurationError(
                    "SharedMemory does not satisfy SimulationAgent's publication contract.",
                    component=self.name,
                    context={"missing_methods": missing},
                )

    # ------------------------------------------------------------------
    # Model registration / dependency injection
    # ------------------------------------------------------------------

    def register_model(
        self,
        model: SimulationModel,
        *,
        replace: Optional[bool] = None,
    ) -> ModelDescriptor:
        """Register an externally constructed Simulation model."""
        return self.model_registry.register(model, replace=replace)

    def unregister_model(self, model_id: str) -> Optional[SimulationModel]:
        """Remove a model registration without closing the model."""
        return self.model_registry.unregister(model_id)

    def registered_models(self) -> tuple[ModelDescriptor, ...]:
        return self.model_registry.descriptors()

    def _resolve_model(
        self,
        request: SimulationAgentRequest,
    ) -> SimulationModel:
        if request.model is not None:
            model = request.model
            if not callable(getattr(model, "transition", None)):
                raise SimulationModelError(
                    "Supplied Simulation model does not expose transition().",
                    context={"model_type": type(model).__name__},
                )
            return model

        if request.model_id is None:
            raise SimulationModelError(
                "SimulationAgent requires either model or model_id.",
                context={"operation": request.operation.value},
            )
        return self.model_registry.get(request.model_id)

    # ------------------------------------------------------------------
    # Public orchestration API
    # ------------------------------------------------------------------

    def simulate(
        self,
        request: SimulationAgentRequest | Mapping[str, Any],
    ) -> SimulationOutcome:
        """Execute one normalized Simulation operation.

        Numerical work is delegated to subsystem components.  This method never
        selects or ranks plans/actions/outcomes.
        """
        if not self.enabled:
            raise BaseConfigurationError(
                "SimulationAgent is disabled by configuration.",
                component=self.name,
                operation="simulate",
            )

        normalized = (
            request
            if isinstance(request, SimulationAgentRequest)
            else SimulationAgentRequest.from_mapping(request)
        )
        self._validate_request(normalized)

        handler = self._dispatch.get(normalized.operation)
        if handler is None:
            raise BaseValidationError(
                "Unsupported SimulationAgent operation.",
                component=self.name,
                context={"operation": normalized.operation.value},
            )

        self._last_operation = normalized.operation.value
        self._publish_event(
            "request_accepted",
            {
                "request_id": normalized.request_id,
                "operation": normalized.operation.value,
                "run_id": (
                    normalized.simulation.run_id
                    if normalized.simulation is not None
                    else None
                ),
            },
        )
        logger.info(
            "Simulation request accepted | request_id=%s | operation=%s",
            normalized.request_id,
            normalized.operation.value,
        )

        try:
            outcome = handler(normalized)
            records = self._record_outcome(normalized, outcome)
            self._publish_outcome(normalized, outcome, records)
            self._completed_operations += 1
            self._last_failure = None
            logger.info(
                "Simulation request completed | request_id=%s | operation=%s",
                normalized.request_id,
                normalized.operation.value,
            )
            return outcome

        except SimulationError as exc:
            self._remember_failure(normalized, exc)
            self._publish_error(normalized, exc)
            logger.warning(
                "Simulation request failed | request_id=%s | operation=%s | error=%s",
                normalized.request_id,
                normalized.operation.value,
                type(exc).__name__,
            )
            raise

        except (BaseValidationError, BaseStateError, BaseConfigurationError):
            raise

        except Exception as exc:
            error = BaseRuntimeError(
                "SimulationAgent orchestration failed.",
                component=self.name,
                operation=normalized.operation.value,
                context={"request_id": normalized.request_id},
                cause=exc,
            )
            self._remember_failure(normalized, error)
            self._publish_error(normalized, error)
            logger.exception(
                "Unexpected SimulationAgent orchestration failure | request_id=%s | operation=%s",
                normalized.request_id,
                normalized.operation.value,
            )
            raise error from exc

    def perform_task(self, task_data: Any) -> dict[str, Any]:
        if isinstance(task_data, SimulationAgentRequest):
            request = task_data
        elif isinstance(task_data, Mapping):
            payload = dict(task_data)
            payload.setdefault("operation", self.default_operation.value)
            request = SimulationAgentRequest.from_mapping(payload)
        else:
            raise BaseValidationError(
                "SimulationAgent task_data must be a mapping or SimulationAgentRequest.",
                component=self.name,
            )
        return self._outcome_to_dict(self.simulate(request))

    def predict(self, state: Any, context: Any = None) -> dict[str, Any]:
        """Compatibility route for BaseAgent/factory prediction-style dispatch."""
        if not isinstance(state, Mapping):
            raise BaseValidationError(
                "SimulationAgent.predict requires a Simulation request mapping.",
                component=self.name,
            )
        payload = dict(state)
        if context is not None and "context" not in payload:
            payload["context"] = context
        return self.perform_task(payload)

    def act(self, task_data: Any, context: Any = None) -> dict[str, Any]:
        """Compatibility route; it simulates supplied actions and never selects one."""
        return self.predict(task_data, context=context)

    def capabilities(self) -> dict[str, Any]:
        return {
            "agent": self.name,
            "operations": tuple(item.value for item in SimulationOperation),
            "registered_models": len(self.model_registry),
            "simulation_memory": True,
            "shared_memory_publication": (
                self.publish_results or self.publish_lifecycle_events
            ),
            "checkpointing_supported": self.supports_checkpointing,
            "planning_or_action_selection": False,
            "optimization": False,
            "causal_discovery": False,
            "formal_verification": False,
            "safety_gating": False,
        }

    # ------------------------------------------------------------------
    # Orchestration validation
    # ------------------------------------------------------------------

    def _validate_request(self, request: SimulationAgentRequest) -> None:
        requires_simulation = {
            SimulationOperation.ROLLOUT,
            SimulationOperation.TRANSITION,
            SimulationOperation.MONTE_CARLO,
            SimulationOperation.SCENARIO,
            SimulationOperation.COUNTERFACTUAL,
            SimulationOperation.PERTURBATION,
        }
        if request.operation in requires_simulation and request.simulation is None:
            raise BaseValidationError(
                f"{request.operation.value} requires a SimulationRequest.",
                component=self.name,
                context={"request_id": request.request_id},
            )

        requires_model = requires_simulation | {SimulationOperation.REPLAY}
        if request.operation in requires_model:
            if request.model is None and request.model_id is None:
                raise BaseValidationError(
                    f"{request.operation.value} requires model or model_id.",
                    component=self.name,
                    context={"request_id": request.request_id},
                )

        if request.operation is SimulationOperation.TRANSITION:
            assert request.simulation is not None
            if len(request.simulation.actions) != 1:
                raise BaseValidationError(
                    "transition requires exactly one externally supplied action.",
                    component=self.name,
                    context={"action_count": len(request.simulation.actions)},
                )

        if request.operation is SimulationOperation.SCENARIO:
            if request.scenario is None and not request.scenarios:
                raise BaseValidationError(
                    "scenario operation requires scenario or scenarios.",
                    component=self.name,
                )
            if request.scenario is not None and request.scenarios:
                raise BaseValidationError(
                    "Supply either scenario or scenarios, not both.",
                    component=self.name,
                )

        if request.operation is SimulationOperation.COUNTERFACTUAL:
            if not request.interventions:
                raise BaseValidationError(
                    "counterfactual requires externally supplied interventions.",
                    component=self.name,
                )

        if request.operation is SimulationOperation.PERTURBATION:
            if not request.parameter_overrides:
                raise BaseValidationError(
                    "perturbation requires parameter_overrides.",
                    component=self.name,
                )

        if request.operation is SimulationOperation.REPLAY and request.source_result is None:
            raise SimulationMemoryError(
                "replay requires source_result=SimulationResult; "
                "SimulationMemory intentionally stores bounded lineage metadata, "
                "not a second typed-result database."
            )

    # ------------------------------------------------------------------
    # Subsystem dispatch
    # ------------------------------------------------------------------

    def _dispatch_rollout(
        self,
        request: SimulationAgentRequest,
    ) -> SimulationResult:
        assert request.simulation is not None
        return self.rollout_engine.rollout(
            request.simulation,
            self._resolve_model(request),
        )

    def _dispatch_transition(
        self,
        request: SimulationAgentRequest,
    ) -> SimulationResult:
        assert request.simulation is not None
        return self.rollout_engine.rollout(
            request.simulation,
            self._resolve_model(request),
        )

    def _dispatch_monte_carlo(
        self,
        request: SimulationAgentRequest,
    ) -> MonteCarloResult:
        assert request.simulation is not None
        options = request.options
        return self.rollout_engine.rollout_many(
            request.simulation,
            self._resolve_model(request),
            sample_count=options.get("sample_count"),
            seed=options.get("seed"),
            concurrency=options.get("concurrency"),
            timeout_seconds=options.get("timeout_seconds"),
            continue_on_error=options.get("continue_on_error"),
            metadata=request.metadata,
        )

    def _dispatch_scenario(
        self,
        request: SimulationAgentRequest,
    ) -> SimulationResult | ScenarioBatchResult:
        assert request.simulation is not None
        model = self._resolve_model(request)
        if request.scenario is not None:
            return self.scenario_engine.run_scenario(
                request.simulation,
                model,
                request.scenario,
            )
        return self.scenario_engine.execute_scenarios(
            request.simulation,
            model,
            request.scenarios,
            metadata=request.metadata,
        )

    def _dispatch_counterfactual(
        self,
        request: SimulationAgentRequest,
    ) -> CounterfactualRollout:
        assert request.simulation is not None
        options = request.options
        return self.rollout_engine.counterfactual_rollout(
            request.simulation,
            self._resolve_model(request),
            request.interventions,
            run_baseline=coerce_bool(options.get("run_baseline", True), True),
            alternate_scenario_id=options.get("alternate_scenario_id"),
            metadata=request.metadata,
        )

    def _dispatch_perturbation(
        self,
        request: SimulationAgentRequest,
    ) -> SimulationResult:
        assert request.simulation is not None
        return self.rollout_engine.perturbed_rollout(
            request.simulation,
            self._resolve_model(request),
            request.parameter_overrides,
            run_id=request.options.get("run_id"),
            scenario_id=request.options.get("scenario_id"),
            seed=request.options.get("seed"),
        )

    def _dispatch_replay(
        self,
        request: SimulationAgentRequest,
    ) -> SimulationResult:
        assert request.source_result is not None
        return self.rollout_engine.replay(
            request.source_result,
            self._resolve_model(request),
        )

    def _dispatch_sensitivity(
        self,
        request: SimulationAgentRequest,
    ) -> SensitivityEstimate:
        method = str(request.options.get("method", "")).strip().lower()
        if method in {"oat", "one_at_a_time"}:
            return self.analysis.oat_effects(
                baseline_output=float(request.options["baseline_output"]),
                perturbed_outputs=self._require_mapping_option(request, "perturbed_outputs"),
                parameter_deltas=self._require_mapping_option(request, "parameter_deltas"),
            )
        if method == "morris":
            return self.analysis.morris_elementary_effects(
                parameter_names=self._require_sequence_option(
                    request, "parameter_names"
                ),
                input_paths=cast(
                    Sequence[Sequence[Sequence[float]]],
                    self._require_sequence_option(request, "input_paths"),
                ),
                output_paths=cast(
                    Sequence[Sequence[float]],
                    self._require_sequence_option(request, "output_paths"),
                ),
                change_tolerance=float(
                    request.options.get("change_tolerance", 1.0e-12)
                ),
            )
        if method == "sobol":
            return self.analysis.sobol_indices(
                y_a=cast(Sequence[float], self._require_sequence_option(request, "y_a")),
                y_b=cast(Sequence[float], self._require_sequence_option(request, "y_b")),
                y_ab=self._require_mapping_option(request, "y_ab"),
            )
        raise BaseValidationError(
            "sensitivity options.method must be one_at_a_time, morris, or sobol.",
            component=self.name,
            context={"method": method},
        )

    def _dispatch_analysis(
        self,
        request: SimulationAgentRequest,
    ) -> Mapping[str, Any]:
        kind = str(request.options.get("kind", "")).strip().lower()

        if kind == "distribution":
            result = self.analysis.summarize_distribution(self._require_sequence_option(request, "values"))
            return {"kind": kind, "summary": result.to_dict()}

        if kind == "convergence":
            result = self.analysis.convergence_diagnostics(self._require_sequence_option(request, "values"))
            return {"kind": kind, "convergence": result.to_dict()}

        if kind == "trajectory_divergence":
            left = request.options.get("left")
            right = request.options.get("right")
            if not isinstance(left, Trajectory) or not isinstance(right, Trajectory):
                raise BaseValidationError("trajectory_divergence requires left/right Trajectory objects.", component=self.name)
            return {
                "kind": kind,
                "divergence": self.analysis.trajectory_divergence(left, right),
            }

        if kind == "branch_frequencies":
            values = self._require_sequence_option(request, "scenario_ids")
            return {
                "kind": kind,
                "frequencies": self.analysis.branch_frequencies(
                    [str(item) for item in values]
                ),
            }

        if kind == "monte_carlo":
            batch = request.options.get("batch")
            extractors = request.options.get("extractors")
            if not isinstance(batch, MonteCarloResult):
                raise BaseValidationError("monte_carlo analysis requires batch=MonteCarloResult.", component=self.name)
            if not isinstance(extractors, Mapping):
                raise BaseValidationError("monte_carlo analysis requires extractors mapping.", component=self.name)
            enriched = self.analysis.summarize_monte_carlo(batch, extractors)
            return {"kind": kind, "batch": enriched.to_dict()}

        raise BaseValidationError(
            "analysis options.kind is unsupported.",
            component=self.name,
            context={"kind": kind},
        )

    @staticmethod
    def _require_mapping_option(request: SimulationAgentRequest, name: str) -> Mapping[str, Any]:
        value = request.options.get(name)
        if not isinstance(value, Mapping):
            raise BaseValidationError(f"options.{name} must be a mapping.", component="SimulationAgent")
        return value

    @staticmethod
    def _require_sequence_option(request: SimulationAgentRequest, name: str) -> Sequence[Any]:
        value = request.options.get(name)
        if not isinstance(value, Sequence) or isinstance(
            value, (str, bytes, bytearray)
        ):
            raise BaseValidationError(f"options.{name} must be a sequence.", component="SimulationAgent")
        return value

    # ------------------------------------------------------------------
    # SimulationMemory ownership
    # ------------------------------------------------------------------

    def _record_result(
        self,
        result: SimulationResult,
        *,
        request: Optional[SimulationRequest],
        parent_run_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> SimulationMemoryRecord:
        record = self.sim_memory.record(
            result,
            request=request,
            parent_run_id=parent_run_id,
            metadata=metadata,
        )
        self._last_run_id = result.run_id
        return record

    def _record_outcome(
        self,
        request: SimulationAgentRequest,
        outcome: SimulationOutcome,
    ) -> tuple[SimulationMemoryRecord, ...]:
        records: list[SimulationMemoryRecord] = []

        if isinstance(outcome, SimulationResult):
            records.append(
                self._record_result(
                    outcome,
                    request=request.simulation,
                    metadata={
                        "agent_request_id": request.request_id,
                        "operation": request.operation.value,
                    },
                )
            )

        elif isinstance(outcome, MonteCarloResult):
            for sample_index, result in enumerate(outcome.results):
                records.append(
                    self._record_result(
                        result,
                        request=request.simulation,
                        metadata={
                            "agent_request_id": request.request_id,
                            "operation": request.operation.value,
                            "batch_id": outcome.batch_id,
                            "sample_index": sample_index,
                        },
                    )
                )

        elif isinstance(outcome, ScenarioBatchResult):
            scenario_by_id = {item.scenario_id: item for item in request.scenarios}
            run_by_scenario = {
                scenario_id: result.run_id
                for scenario_id, result in outcome.results.items()
            }
            for scenario_id, result in outcome.results.items():
                scenario = scenario_by_id.get(scenario_id)
                parent_run_id = None
                if scenario is not None and scenario.parent_scenario_id is not None:
                    parent_run_id = run_by_scenario.get(scenario.parent_scenario_id)
                records.append(
                    self._record_result(
                        result,
                        request=request.simulation,
                        parent_run_id=parent_run_id,
                        metadata={
                            "agent_request_id": request.request_id,
                            "operation": request.operation.value,
                            "batch_id": outcome.batch_id,
                        },
                    )
                )

        elif isinstance(outcome, CounterfactualRollout):
            baseline_run_id: Optional[str] = None
            if outcome.baseline is not None:
                baseline_record = self._record_result(
                    outcome.baseline,
                    request=request.simulation,
                    metadata={
                        "agent_request_id": request.request_id,
                        "operation": "counterfactual_baseline",
                    },
                )
                baseline_run_id = baseline_record.run_id
                records.append(baseline_record)

            records.append(
                self._record_result(
                    outcome.alternate,
                    request=request.simulation,
                    parent_run_id=baseline_run_id,
                    metadata={
                        "agent_request_id": request.request_id,
                        "operation": "counterfactual_alternate",
                    },
                )
            )

        return tuple(records)

    def recent_runs(
        self,
        limit: Optional[int] = None,
    ) -> tuple[SimulationMemoryRecord, ...]:
        return self.sim_memory.recent(limit)

    def run_lineage(
        self,
        run_id: str,
    ) -> tuple[SimulationMemoryRecord, ...]:
        return self.sim_memory.lineage(run_id)

    # ------------------------------------------------------------------
    # SharedMemory: bounded communication only
    # ------------------------------------------------------------------

    def _shared_set(self, key: str, value: Any, *, tags: Sequence[str] = ()) -> bool:
        if not self.publish_results and "error" not in tags:
            return False

        ttl = (
            None
            if self.shared_memory_ttl_seconds <= 0
            else self.shared_memory_ttl_seconds
        )
        payload = to_json_safe(
            value,
            max_depth=10,
            max_items=self.max_shared_list_items,
            max_string_length=self.max_shared_string_length,
        )
        try:
            self.shared_memory.set(
                key,
                payload,
                ttl=ttl,
                tags=list(tags),
                metadata={
                    "agent": self.name,
                    "agent_id": self.agent_id,
                    "schema": "simulation_agent.v1",
                },
            )
        except Exception as exc:
            self._handle_shared_memory_error("set", key, exc)
            return False

        self._mark_runtime_recovered("communication", "shared_memory.set")
        return True

    def _publish_event(self, event: str, payload: Mapping[str, Any]) -> None:
        if not self.publish_lifecycle_events:
            return
        message = {
            "schema": "simulation_agent.event.v1",
            "event": str(event),
            "agent": self.name,
            "agent_id": self.agent_id,
            "timestamp": utc_now_iso(),
            "payload": to_json_safe(
                dict(payload),
                max_depth=8,
                max_items=self.max_shared_list_items,
                max_string_length=self.max_shared_string_length,
            ),
        }
        try:
            self.shared_memory.publish(self.event_channel, message)
        except Exception as exc:
            self._handle_shared_memory_error("publish", self.event_channel, exc)
            return
        self._mark_runtime_recovered("communication", "shared_memory.publish")

    def _handle_shared_memory_error(self, operation: str, key: str, exc: BaseException) -> None:
        self._mark_runtime_degraded(
            "communication",
            f"shared_memory.{operation}",
            exc,
        )
        logger.warning(
            "SimulationAgent SharedMemory %s degraded | key=%s | error=%s",
            operation,
            key,
            type(exc).__name__,
        )

    def _publish_outcome(
        self,
        request: SimulationAgentRequest,
        outcome: SimulationOutcome,
        records: Sequence[SimulationMemoryRecord],
    ) -> None:
        if not self.publish_results:
            self._publish_event(
                "completed",
                {
                    "request_id": request.request_id,
                    "operation": request.operation.value,
                    "run_ids": [record.run_id for record in records],
                },
            )
            return

        summary = self._summary_payload(request, outcome, records)
        result_key = f"{self.result_key_prefix}:{request.request_id}"
        summary_key = f"{self.summary_key_prefix}:{request.request_id}"

        self._shared_set(
            result_key,
            self._bounded_result_payload(outcome),
            tags=("simulation", "result", request.operation.value),
        )
        self._shared_set(
            summary_key,
            summary,
            tags=("simulation", "summary", request.operation.value),
        )
        self._shared_set(
            self.latest_result_key,
            summary,
            tags=("simulation", "latest"),
        )
        self._publish_event("completed", summary)

    def _publish_error(self, request: SimulationAgentRequest, error: BaseException) -> None:
        if self.publish_errors:
            payload = {
                "schema": "simulation_agent.error.v1",
                "request_id": request.request_id,
                "operation": request.operation.value,
                "error_type": type(error).__name__,
                "message": str(error)[: self.max_shared_string_length],
                "timestamp": utc_now_iso(),
            }
            self._shared_set(
                f"{self.error_key_prefix}:{request.request_id}",
                payload,
                tags=("simulation", "error", request.operation.value),
            )
        self._publish_event(
            "failed",
            {
                "request_id": request.request_id,
                "operation": request.operation.value,
                "error_type": type(error).__name__,
            },
        )

    def _summary_payload(
        self,
        request: SimulationAgentRequest,
        outcome: SimulationOutcome,
        records: Sequence[SimulationMemoryRecord],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": "simulation_agent.summary.v1",
            "request_id": request.request_id,
            "operation": request.operation.value,
            "run_ids": [record.run_id for record in records][
                : self.max_shared_list_items
            ],
            "simulation_memory_references": [record.run_id for record in records][
                : self.max_shared_list_items
            ],
            "record_count": len(records),
            "timestamp": utc_now_iso(),
        }

        if isinstance(outcome, SimulationResult):
            payload.update(self._simulation_result_summary(outcome))
        elif isinstance(outcome, MonteCarloResult):
            payload.update(
                {
                    "batch_id": outcome.batch_id,
                    "root_seed": outcome.root_seed,
                    "sample_count": outcome.sample_count,
                    "completed_samples": len(outcome.results),
                    "failed_samples": len(outcome.failures),
                    "summary_fields": tuple(outcome.summaries.keys()),
                    "convergence_fields": tuple(outcome.convergence.keys()),
                }
            )
        elif isinstance(outcome, ScenarioBatchResult):
            payload.update(
                {
                    "batch_id": outcome.batch_id,
                    "scenario_count": len(outcome.results),
                    "scenario_ids": tuple(outcome.results.keys())[
                        : self.max_shared_list_items
                    ],
                    "warning_count": len(outcome.warnings),
                }
            )
        elif isinstance(outcome, CounterfactualRollout):
            payload.update(
                {
                    "baseline_run_id": (
                        outcome.baseline.run_id
                        if outcome.baseline is not None
                        else None
                    ),
                    "alternate_run_id": outcome.alternate.run_id,
                    "intervention_ids": tuple(
                        item.intervention_id for item in outcome.interventions
                    )[: self.max_shared_list_items],
                }
            )
        elif isinstance(outcome, SensitivityEstimate):
            payload.update(
                {
                    "method": outcome.method,
                    "first_order_parameters": tuple(outcome.first_order.keys())[
                        : self.max_shared_list_items
                    ],
                    "total_order_parameters": tuple(outcome.total_order.keys())[
                        : self.max_shared_list_items
                    ],
                    "elementary_effect_parameters": tuple(
                        outcome.elementary_mean.keys()
                    )[: self.max_shared_list_items],
                }
            )
        else:
            payload["result_kind"] = "analysis"
        return payload

    def _bounded_result_payload(self, outcome: SimulationOutcome) -> Mapping[str, Any]:
        if isinstance(outcome, SimulationResult):
            payload = self._simulation_result_summary(outcome)
            if self.include_trajectory_in_shared_memory:
                steps = outcome.trajectory.steps[: self.max_shared_trajectory_steps]
                payload["trajectory"] = {
                    "trajectory_id": outcome.trajectory.trajectory_id,
                    "termination_reason": outcome.termination_reason.value,
                    "transition_count": outcome.trajectory.transition_count,
                    "steps": [step.to_dict() for step in steps],
                    "truncated": (
                        len(outcome.trajectory.steps)
                        > self.max_shared_trajectory_steps
                    ),
                }
            return payload

        if isinstance(outcome, MonteCarloResult):
            return {
                "batch_id": outcome.batch_id,
                "root_seed": outcome.root_seed,
                "sample_count": outcome.sample_count,
                "completed_samples": len(outcome.results),
                "failed_samples": len(outcome.failures),
                "summaries": {
                    name: value.to_dict()
                    for name, value in list(outcome.summaries.items())[
                        : self.max_shared_list_items
                    ]
                },
                "convergence": {
                    name: value.to_dict()
                    for name, value in list(outcome.convergence.items())[
                        : self.max_shared_list_items
                    ]
                },
            }

        if isinstance(outcome, ScenarioBatchResult):
            return {
                "batch_id": outcome.batch_id,
                "scenario_count": len(outcome.results),
                "results": {
                    scenario_id: self._simulation_result_summary(result)
                    for scenario_id, result in list(outcome.results.items())[
                        : self.max_shared_list_items
                    ]
                },
                "branches": {
                    scenario_id: branch.to_dict()
                    for scenario_id, branch in list(outcome.branches.items())[
                        : self.max_shared_list_items
                    ]
                },
                "warnings": list(outcome.warnings)[
                    : self.max_shared_list_items
                ],
            }

        if isinstance(outcome, CounterfactualRollout):
            return {
                "baseline": (
                    None
                    if outcome.baseline is None
                    else self._simulation_result_summary(outcome.baseline)
                ),
                "alternate": self._simulation_result_summary(outcome.alternate),
                "interventions": [
                    item.to_dict()
                    for item in outcome.interventions[
                        : self.max_shared_list_items
                    ]
                ],
                "metadata": dict(outcome.metadata),
            }

        if isinstance(outcome, SensitivityEstimate):
            return outcome.to_dict()

        return dict(outcome)

    @staticmethod
    def _simulation_result_summary(result: SimulationResult) -> dict[str, Any]:
        return {
            "run_id": result.run_id,
            "scenario_id": result.scenario_id,
            "model_id": result.model_id,
            "model_version": result.model_version,
            "trajectory_id": result.trajectory.trajectory_id,
            "seed": result.seed,
            "termination_reason": result.termination_reason.value,
            "transition_count": result.trajectory.transition_count,
            "duration": result.trajectory.duration,
            "runtime_seconds": result.runtime.duration_seconds,
            "deterministic": result.runtime.deterministic,
            "warning_count": len(result.warnings),
            "result_fingerprint": stable_fingerprint(
                {
                    "run_id": result.run_id,
                    "model_id": result.model_id,
                    "model_version": result.model_version,
                    "seed": result.seed,
                    "terminal_state": result.terminal_state,
                    "termination_reason": result.termination_reason.value,
                },
                length=32,
            ),
        }

    @staticmethod
    def _outcome_to_dict(outcome: SimulationOutcome) -> dict[str, Any]:
        if isinstance(outcome, Mapping):
            return dict(outcome)
        to_dict = getattr(outcome, "to_dict", None)
        if callable(to_dict):
            value = to_dict()
            if isinstance(value, Mapping):
                return dict(value)
        raise BaseRuntimeError(
            "SimulationAgent outcome does not expose a mapping representation.",
            component="SimulationAgent",
            context={"outcome_type": type(outcome).__name__},
        )

    # ------------------------------------------------------------------
    # Health / diagnostics
    # ------------------------------------------------------------------

    def _remember_failure(
        self,
        request: SimulationAgentRequest,
        error: BaseException,
    ) -> None:
        self._last_failure = {
            "request_id": request.request_id,
            "operation": request.operation.value,
            "error_type": type(error).__name__,
            "message": str(error)[:512],
            "timestamp": utc_now_iso(),
        }

    def health_check(self) -> dict[str, Any]:
        memory_stats = self.sim_memory.stats()
        descriptors = self.model_registry.descriptors()
        runtime = self.runtime_status()

        return {
            "status": runtime.get("health", "healthy"),
            "agent": self.name,
            "enabled": self.enabled,
            "runtime": runtime,
            "simulation_subsystem": {
                "rollout_engine": True,
                "scenario_engine": True,
                "analysis": True,
                "model_registry": True,
            },
            "simulation_memory": memory_stats,
            "registered_models": [
                descriptor.to_dict()
                for descriptor in descriptors[: self.max_shared_list_items]
            ],
            "registered_model_count": len(descriptors),
            "last_operation": self._last_operation,
            "last_run_id": self._last_run_id,
            "last_failure": self._last_failure,
            "completed_operations": self._completed_operations,
            "shared_memory_publication": {
                "publish_results": self.publish_results,
                "publish_lifecycle_events": self.publish_lifecycle_events,
                "publish_errors": self.publish_errors,
                "event_channel": self.event_channel,
            },
            "checkpointing": {
                "supported": self.supports_checkpointing,
                "manager_injected": self.checkpoint_manager is not None,
                "enabled": self.checkpointing_enabled,
                "reason": (
                    None
                    if self.supports_checkpointing
                    else "SimulationMemory has no public typed-result restore/import contract."
                ),
            },
        }


__all__ = [
    "SimulationAgent",
    "SimulationAgentRequest",
    "SimulationOperation",
    "SimulationOutcome",
]
