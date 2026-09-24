"""
SLAI Verification Agent. Top-level orchestration facade for the formal Verification subsystem.

Implements:
- Baier & Katoen (2008), Principles of Model Checking.
- Clarke, Grumberg, Kroening & Peled (2018), Model Checking, 2nd ed.
- Clarke, Henzinger, Veith & Bloem (2018), Handbook of Model Checking.
- Hoare (1969), An Axiomatic Basis for Computer Programming.
- Cousot & Cousot (1977), Abstract Interpretation.
- Pnueli (1977), The Temporal Logic of Programs.
- Biere, Heule, van Maaren & Walsh (2021), Handbook of Satisfiability, 2nd ed.
- Barrett, Sebastiani, Seshia & Tinelli (2021), Satisfiability Modulo Theories.
- de Moura & Bjørner (2008), Z3: An Efficient SMT Solver

VerificationAgent establishes or refutes formally stated properties over explicit formal models and constraints,
producing verifiable evidence where possible.
"""

from __future__ import annotations

__version__ = "2.3.0"

import math
import uuid

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeAlias, cast

from .base_agent import BaseAgent
from .base.utils.base_errors import *
from .base.utils.base_helpers import *
from .base.utils.config_contract import assert_valid_config_contract
from .base.utils.main_config_loader import *
from .verification import *
from .verification.utils.verification_errors import *
from .verification.utils.verification_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("Verification Agent")
printer = PrettyPrinter()


class VerificationOperation(str, Enum):
    """Formal procedures currently exposed by the completed subsystem."""

    SATISFIABILITY = "satisfiability"
    VALIDITY = "validity"
    INVARIANT = "invariant"
    INDUCTIVE_INVARIANT = "inductive_invariant"
    REACHABILITY = "reachability"
    ABSTRACT_FIXPOINT = "abstract_fixpoint"
    ABSTRACT_INVARIANT = "abstract_invariant"


FormalResult: TypeAlias = VerificationResult[object] | FixpointResult[object] | AbstractInvariantCheck[object]


@dataclass(frozen=True, slots=True)
class VerificationAgentRequest:
    """Normalized agent-facing formal verification request.

    ``artifact`` and ``formal_property`` intentionally retain the subsystem's
    native typed objects instead of introducing duplicate agent-level model or
    formula classes. Their admissible runtime types depend on ``operation``.
    """

    operation: VerificationOperation
    artifact: object
    formal_property: object | None = None
    assumptions: tuple[NamedConstraint, ...] = ()
    bounds: ResourceBounds | None = None
    backend_name: str | None = None
    property_name: str | None = None
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    options: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, str | int | float | bool | None] = field(default_factory=dict)
    prior_request_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        operation = self.operation
        if not isinstance(operation, VerificationOperation):
            try:
                operation = VerificationOperation(str(operation).strip().lower())
            except (TypeError, ValueError) as exc:
                raise UnsupportedVerificationError(
                    "unsupported VerificationAgent operation",
                    context={"operation": str(self.operation)},
                    cause=exc,
                ) from exc
            object.__setattr__(self, "operation", operation)

        request_id = str(self.request_id or "").strip()
        if not request_id:
            raise MalformedSpecificationError("request_id must be a non-empty string")
        object.__setattr__(self, "request_id", request_id)

        assumptions = tuple(self.assumptions)
        if any(not isinstance(item, NamedConstraint) for item in assumptions):
            raise MalformedSpecificationError(
                "assumptions must contain NamedConstraint objects"
            )
        object.__setattr__(self, "assumptions", assumptions)

        if self.bounds is not None and not isinstance(self.bounds, ResourceBounds):
            raise MalformedSpecificationError("bounds must be ResourceBounds or null")

        backend_name = None if self.backend_name is None else str(self.backend_name).strip().lower()
        if backend_name == "":
            backend_name = None
        object.__setattr__(self, "backend_name", backend_name)

        property_name = None if self.property_name is None else str(self.property_name).strip()
        if property_name == "":
            property_name = None
        object.__setattr__(self, "property_name", property_name)

        if not isinstance(self.options, Mapping):
            raise MalformedSpecificationError("options must be a mapping")
        object.__setattr__(self, "options", dict(self.options))

        if not isinstance(self.metadata, Mapping):
            raise MalformedSpecificationError("metadata must be a mapping")
        normalized_metadata: dict[str, str | int | float | bool | None] = {}
        for raw_key, value in self.metadata.items():
            key = str(raw_key).strip()
            if not key:
                raise MalformedSpecificationError("metadata keys must be non-empty strings")
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise MalformedSpecificationError(
                    "metadata values must be scalar",
                    context={"key": key, "type": type(value).__name__},
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise MalformedSpecificationError(
                    "metadata floating-point values must be finite",
                    context={"key": key, "value": repr(value)},
                )
            normalized_metadata[key] = value
        object.__setattr__(self, "metadata", normalized_metadata)

        prior = tuple(str(item).strip() for item in self.prior_request_ids)
        if any(not item for item in prior):
            raise MalformedSpecificationError("prior_request_ids must contain non-empty strings")
        # Stable de-duplication keeps shared-memory references deterministic.
        object.__setattr__(self, "prior_request_ids", tuple(dict.fromkeys(prior)))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "VerificationAgentRequest":
        if not isinstance(payload, Mapping):
            raise MalformedSpecificationError("VerificationAgent request must be a mapping")
        raw_operation = payload.get("operation")
        if raw_operation is None:
            raise MalformedSpecificationError("VerificationAgent request requires operation")
        try:
            operation = raw_operation if isinstance(raw_operation, VerificationOperation) else VerificationOperation(str(raw_operation).strip().lower())
        except (TypeError, ValueError) as exc:
            raise UnsupportedVerificationError(
                "unsupported VerificationAgent operation",
                context={"operation": str(raw_operation)},
                cause=exc,
            ) from exc

        if "artifact" not in payload:
            raise MalformedSpecificationError("VerificationAgent request requires artifact")

        raw_assumptions = payload.get("assumptions", ())
        if raw_assumptions is None:
            assumptions: tuple[NamedConstraint, ...] = ()
        elif isinstance(raw_assumptions, Sequence) and not isinstance(raw_assumptions, (str, bytes, bytearray)):
            assumptions = tuple(raw_assumptions)  # type: ignore[arg-type]
        else:
            raise MalformedSpecificationError("assumptions must be a sequence of NamedConstraint objects")

        raw_prior = payload.get("prior_request_ids", ())
        if raw_prior is None:
            prior: tuple[str, ...] = ()
        elif isinstance(raw_prior, str):
            prior = (raw_prior,)
        elif isinstance(raw_prior, Sequence):
            prior = tuple(str(item) for item in raw_prior)
        else:
            raise MalformedSpecificationError("prior_request_ids must be a sequence of strings")

        options = payload.get("options", {})
        metadata = payload.get("metadata", {})
        if not isinstance(options, Mapping):
            raise MalformedSpecificationError("options must be a mapping")
        if not isinstance(metadata, Mapping):
            raise MalformedSpecificationError("metadata must be a mapping")

        formal_property = payload.get("formal_property", payload.get("property"))
        request_id = str(payload.get("request_id") or uuid.uuid4().hex)

        return cls(
            operation=operation,
            artifact=payload["artifact"],
            formal_property=formal_property,
            assumptions=assumptions,
            bounds=payload.get("bounds"),  # type: ignore[arg-type]
            backend_name=payload.get("backend_name"),  # type: ignore[arg-type]
            property_name=payload.get("property_name"),  # type: ignore[arg-type]
            request_id=request_id,
            options=dict(options),
            metadata=dict(metadata),  # type: ignore[arg-type]
            prior_request_ids=prior,
        )


@dataclass(frozen=True, slots=True)
class VerificationAgentOutcome:
    """Execution envelope preserving formal semantics separately from execution success."""

    request_id: str
    operation: VerificationOperation
    result: FormalResult
    prior_references: tuple[Mapping[str, object], ...] = ()

    @property
    def verification_status(self) -> VerificationStatus | None:
        return self.result.status if isinstance(self.result, VerificationResult) else None

    @property
    def execution_success(self) -> bool:
        # Reaching this object means the formal procedure completed without an
        # operational exception. REFUTED/UNSAT/etc. remain legitimate outcomes.
        return True

    def formal_result_dict(self) -> dict[str, object]:
        result = self.result
        if isinstance(result, VerificationResult):
            return dict(result.to_dict())
        if isinstance(result, FixpointResult):
            return {
                "kind": "abstract_fixpoint",
                "value": to_json_safe(result.value, max_depth=8, max_items=1_000, max_string_length=8_192),
                "converged": result.converged,
                "iterations": result.iterations,
                "used_widening": result.used_widening,
                "reason": result.reason,
            }
        return {
            "kind": "abstract_invariant",
            "candidate": to_json_safe(result.candidate, max_depth=8, max_items=1_000, max_string_length=8_192),
            "post_fixpoint": result.post_fixpoint,
            "initial_included": result.initial_included,
            "inductive": result.inductive,
        }

    def to_dict(self) -> dict[str, object]:
        status = self.verification_status
        return {
            "execution_success": True,
            "request_id": self.request_id,
            "operation": self.operation.value,
            "verification_status": status.value if status is not None else None,
            "formal_result": self.formal_result_dict(),
            "prior_references": [dict(item) for item in self.prior_references],
        }


class VerificationAgent(BaseAgent):
    """Production orchestration facade over ``src.agents.verification``.

    The facade intentionally has no local verification memory and no solver
    implementation. Cross-agent state is published through BaseAgent's injected
    SharedMemory; formal procedures are delegated to the existing subsystem.
    """

    CHECKPOINTING_SUPPORTED = False

    def __init__(
        self,
        shared_memory: object,
        agent_factory: object,
        config: Mapping[str, object] | None = None,
        *,
        checkpoint_manager: object | None = None,
    ) -> None:
        # Agent-level overrides are deliberately not passed to BaseAgent. BaseAgent
        # owns its own base_agent configuration section; VerificationAgent owns
        # only the verification_agent section below.
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            checkpoint_manager=checkpoint_manager,
        )

        self.agent_config: dict[str, object] = dict(get_config_section("verification_agent") or {})
        if config is not None:
            if not isinstance(config, Mapping):
                raise BaseConfigurationError(
                    "VerificationAgent config override must be a mapping",
                    component=self.name,
                )
            self.agent_config.update(dict(config))

        self._load_agent_config()
        self._validate_agent_config()

        # These are stateless orchestration services. Backend discovery/imports
        # remain inside the subsystem and are lazy/optional there.
        self._model_checker = ModelChecker()
        self._satisfiability = SatisfiabilityVerifier()

        logger.info(
            "Verification Agent initialized | shared_memory=%s | publish_evidence=%s",
            self.publish_shared_memory,
            self.publish_evidence_metadata,
        )
        self._publish_event("initialized", {"agent_id": self.agent_id})

    # ------------------------------------------------------------------
    # Agent-level configuration: agents_config.yaml only
    # ------------------------------------------------------------------

    def _cfg(self, key: str, default: object) -> object:
        return self.agent_config.get(key, default)

    def _load_agent_config(self) -> None:
        self.enabled = coerce_bool(self._cfg("enabled", True), True)
        self.publish_shared_memory = coerce_bool(self._cfg("publish_shared_memory", True), True)
        self.fail_on_shared_memory_error = coerce_bool(self._cfg("fail_on_shared_memory_error", False), False)
        self.publish_evidence_metadata = coerce_bool(self._cfg("publish_evidence_metadata", True), True)
        self.shared_memory_ttl_seconds = coerce_int(
            self._cfg("shared_memory_ttl_seconds", 604800), 604800, minimum=0, maximum=31_536_000
        )
        self.max_shared_trace_steps = coerce_int(
            self._cfg("max_shared_trace_steps", 64), 64, minimum=1, maximum=10_000
        )
        self.max_shared_model_entries = coerce_int(
            self._cfg("max_shared_model_entries", 128), 128, minimum=1, maximum=100_000
        )
        self.max_shared_list_items = coerce_int(
            self._cfg("max_shared_list_items", 128), 128, minimum=1, maximum=100_000
        )
        self.max_prior_references = coerce_int(
            self._cfg("max_prior_references", 32), 32, minimum=0, maximum=1_000
        )

        self.request_key_prefix = str(self._cfg("request_key_prefix", "verification_agent.request") or "").strip()
        self.result_key_prefix = str(self._cfg("result_key_prefix", "verification_agent.result") or "").strip()
        self.summary_key_prefix = str(self._cfg("summary_key_prefix", "verification_agent.summary") or "").strip()
        self.error_key_prefix = str(self._cfg("error_key_prefix", "verification_agent.error") or "").strip()
        self.event_key_prefix = str(self._cfg("event_key_prefix", "verification_agent.event") or "").strip()
        self.latest_result_key = str(self._cfg("latest_result_key", "verification_agent.latest") or "").strip()

    def _validate_agent_config(self) -> None:
        for field_name in (
            "request_key_prefix",
            "result_key_prefix",
            "summary_key_prefix",
            "error_key_prefix",
            "event_key_prefix",
            "latest_result_key",
        ):
            if not getattr(self, field_name):
                raise BaseConfigurationError(
                    f"verification_agent.{field_name} must be a non-empty string",
                    component=self.name,
                    context={"field": field_name},
                )
        if self.publish_shared_memory:
            missing = [
                name
                for name in ("get", "set")
                if not callable(getattr(self.shared_memory, name, None))
            ]
            if missing:
                raise BaseConfigurationError(
                    "SharedMemory does not satisfy VerificationAgent's get/set contract",
                    component=self.name,
                    context={"missing_methods": missing},
                )

    # ------------------------------------------------------------------
    # Public facade API
    # ------------------------------------------------------------------

    def verify(self, request: VerificationAgentRequest | Mapping[str, object]) -> VerificationAgentOutcome:
        """Run one formal verification request through the existing subsystem."""

        if not self.enabled:
            raise BaseConfigurationError(
                "VerificationAgent is disabled by configuration",
                component=self.name,
                operation="verify",
            )
        normalized = request if isinstance(request, VerificationAgentRequest) else VerificationAgentRequest.from_mapping(request)
        prior_references = self._load_prior_references(normalized.prior_request_ids)
        request_metadata = self._request_metadata(normalized, prior_references)

        logger.info(
            "Verification request received | id=%s | operation=%s | property=%s | backend=%s",
            normalized.request_id,
            normalized.operation.value,
            normalized.property_name or self._default_property_name(normalized),
            normalized.backend_name or "auto",
        )
        self._shared_set(
            self._key(self.request_key_prefix, normalized.request_id),
            request_metadata,
            tags=("verification", "request", normalized.operation.value),
        )
        self._publish_event(
            "request_received",
            {
                "request_id": normalized.request_id,
                "operation": normalized.operation.value,
                "property_name": normalized.property_name or self._default_property_name(normalized),
            },
        )

        try:
            formal_result = self._dispatch(normalized)
            outcome = VerificationAgentOutcome(
                request_id=normalized.request_id,
                operation=normalized.operation,
                result=formal_result,
                prior_references=prior_references,
            )
            self._publish_outcome(normalized, outcome)
            status = outcome.verification_status
            logger.info(
                "Verification request completed | id=%s | operation=%s | status=%s | evidence=%s",
                normalized.request_id,
                normalized.operation.value,
                status.value if status is not None else self._non_status_label(formal_result),
                self._evidence_available(formal_result),
            )
            return outcome
        except VerificationError as exc:
            self._publish_error(normalized, exc)
            logger.warning(
                "Verification request failed operationally | id=%s | operation=%s | error=%s",
                normalized.request_id,
                normalized.operation.value,
                type(exc).__name__,
            )
            raise
        except Exception as exc:
            error = VerificationError(
                "VerificationAgent orchestration failed",
                component=self.name,
                operation=normalized.operation.value,
                context={"request_id": normalized.request_id},
                cause=exc,
            )
            self._publish_error(normalized, error)
            logger.exception(
                "VerificationAgent orchestration failure | id=%s | operation=%s",
                normalized.request_id,
                normalized.operation.value,
            )
            raise error from exc

    def perform_task(self, task_data: object) -> dict[str, object]:
        """BaseAgent task entry point; formal outcomes remain distinct from execution success."""

        if isinstance(task_data, VerificationAgentRequest):
            request = task_data
        elif isinstance(task_data, Mapping):
            request = VerificationAgentRequest.from_mapping(task_data)
        else:
            raise MalformedSpecificationError(
                "VerificationAgent task_data must be a request mapping or VerificationAgentRequest"
            )
        return self.verify(request).to_dict()

    def predict(self, state: object, context: object = None) -> dict[str, object]:
        """Compatibility route for BaseAgent/factory predict-style dispatch."""

        if isinstance(state, Mapping):
            payload = dict(state)
            if context is not None and "context" not in payload:
                payload["context"] = context
            return self.perform_task(payload)
        raise MalformedSpecificationError("VerificationAgent.predict requires a request mapping")

    def act(self, task_data: object, context: object = None) -> dict[str, object]:
        """Compatibility route for BaseAgent/factory action-style dispatch."""

        return self.predict(task_data, context=context)

    def get_verification_result(self, request_id: str) -> Mapping[str, object] | None:
        """Return an Agent-published result from SharedMemory, if present."""

        if not self.publish_shared_memory:
            return None
        key = self._key(self.result_key_prefix, str(request_id).strip())
        try:
            value = self.shared_memory.get(key)
        except Exception as exc:
            self._handle_shared_memory_error("get", key, exc)
            return None
        return value if isinstance(value, Mapping) else None

    def capabilities(self) -> dict[str, object]:
        return {
            "agent": self.name,
            "operations": tuple(item.value for item in VerificationOperation),
            "formal_statuses": tuple(item.value for item in VerificationStatus),
            "shared_memory": self.publish_shared_memory,
            "optional_solver_imports_in_agent": False,
        }

    # ------------------------------------------------------------------
    # Subsystem dispatch -- orchestration only
    # ------------------------------------------------------------------

    def _dispatch(self, request: VerificationAgentRequest) -> FormalResult:
        operation = request.operation
        property_name = request.property_name or self._default_property_name(request)

        logger.debug(
            "Dispatching verification subsystem | id=%s | operation=%s",
            request.request_id,
            operation.value,
        )

        if operation is VerificationOperation.SATISFIABILITY:
            if not isinstance(request.artifact, SolverRequest):
                raise MalformedSpecificationError("satisfiability requires artifact=SolverRequest")
            if request.assumptions:
                raise MalformedSpecificationError(
                    "satisfiability assumptions must be carried by SolverRequest, not duplicated at agent level"
                )
            return self._satisfiability.check(
                request.artifact,
                backend_name=request.backend_name,
                property_name=property_name,
            )

        if operation is VerificationOperation.VALIDITY:
            if not isinstance(request.artifact, Term):
                raise MalformedSpecificationError("validity requires artifact=Term")
            timeout_seconds = request.options.get("timeout_seconds")
            if timeout_seconds is not None:
                try:
                    timeout_seconds = float(cast(Any, timeout_seconds))
                except (TypeError, ValueError, OverflowError) as exc:
                    raise MalformedSpecificationError("timeout_seconds must be numeric or null", cause=exc) from exc
                if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
                    raise MalformedSpecificationError("timeout_seconds must be finite and greater than zero")
            return self._satisfiability.prove(
                request.artifact,
                assumptions=request.assumptions,
                backend_name=request.backend_name,
                property_name=property_name,
                timeout_seconds=timeout_seconds,  # type: ignore[arg-type]
            )

        if operation is VerificationOperation.INVARIANT:
            system, invariant = self._require_transition_property(request, Invariant, "invariant")
            return self._model_checker.check_invariant(
                system,
                cast(Invariant[object], invariant),
                bounds=request.bounds,
            )

        if operation is VerificationOperation.INDUCTIVE_INVARIANT:
            system, invariant = self._require_transition_property(request, Invariant, "inductive_invariant")
            return self._model_checker.check_inductive_invariant(
                system,
                cast(Invariant[object], invariant),
                bounds=request.bounds,
            )

        if operation is VerificationOperation.REACHABILITY:
            system, target = self._require_transition_property(request, StatePredicate, "reachability")
            return self._model_checker.check_reachability(system, target, bounds=request.bounds)

        if operation is VerificationOperation.ABSTRACT_FIXPOINT:
            if not isinstance(request.artifact, AbstractDomain):
                raise MalformedSpecificationError("abstract_fixpoint requires artifact implementing AbstractDomain")
            transfer = self._require_callable_property(request, "abstract_fixpoint")
            max_iterations = self._option_int(request, "max_iterations", 256, minimum=1)
            widening_after_raw = request.options.get("widening_after")
            widening_after = None if widening_after_raw is None else self._coerce_option_int(
                widening_after_raw, "widening_after", minimum=0
            )
            widening = request.options.get("widening")
            if widening is not None and not callable(widening):
                raise MalformedSpecificationError("widening must be callable or null")
            return compute_post_fixpoint(
                request.artifact,
                transfer,
                initial=request.options.get("initial"),
                max_iterations=max_iterations,
                widening_after=widening_after,
                widening=widening,  # type: ignore[arg-type]
            )

        if operation is VerificationOperation.ABSTRACT_INVARIANT:
            if not isinstance(request.artifact, AbstractDomain):
                raise MalformedSpecificationError("abstract_invariant requires artifact implementing AbstractDomain")
            transfer = self._require_callable_property(request, "abstract_invariant")
            if "candidate" not in request.options:
                raise MalformedSpecificationError("abstract_invariant requires options.candidate")
            return check_abstract_invariant(
                request.artifact,
                transfer,
                request.options["candidate"],
                initial=request.options.get("initial"),
            )

        # Enum construction should make this unreachable, but keep the boundary explicit.
        raise UnsupportedVerificationError(
            "unsupported VerificationAgent operation",
            context={"operation": str(operation)},
        )

    @staticmethod
    def _require_transition_property(
        request: VerificationAgentRequest,
        property_type: type[StatePredicate[object]],
        operation: str,
    ) -> tuple[TransitionSystem[object], StatePredicate[object]]:
        if not isinstance(request.artifact, TransitionSystem):
            raise MalformedSpecificationError(f"{operation} requires artifact=TransitionSystem")
        if not isinstance(request.formal_property, property_type):
            raise MalformedSpecificationError(
                f"{operation} requires formal_property={property_type.__name__}"
            )
        return request.artifact, request.formal_property

    @staticmethod
    def _require_callable_property(
        request: VerificationAgentRequest,
        operation: str,
    ) -> Callable[[object], object]:
        if not callable(request.formal_property):
            raise MalformedSpecificationError(f"{operation} requires a callable formal_property transfer")
        return request.formal_property

    @staticmethod
    def _coerce_option_int(value: object, field_name: str, *, minimum: int) -> int:
        if isinstance(value, bool):
            raise MalformedSpecificationError(f"{field_name} must be an integer")
        try:
            parsed = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError, OverflowError) as exc:
            raise MalformedSpecificationError(f"{field_name} must be an integer", cause=exc) from exc
        if parsed < minimum:
            raise MalformedSpecificationError(
                f"{field_name} must be >= {minimum}",
                context={"field": field_name, "value": parsed},
            )
        return parsed

    def _option_int(self, request: VerificationAgentRequest, field_name: str, default: int, *, minimum: int) -> int:
        return self._coerce_option_int(request.options.get(field_name, default), field_name, minimum=minimum)

    @staticmethod
    def _default_property_name(request: VerificationAgentRequest) -> str:
        formal_property = request.formal_property
        name = getattr(formal_property, "name", None)
        if isinstance(name, str) and name.strip():
            return name.strip()
        if request.operation is VerificationOperation.SATISFIABILITY:
            return "constraint_system"
        if request.operation is VerificationOperation.VALIDITY:
            return "property"
        return request.operation.value

    # ------------------------------------------------------------------
    # SharedMemory coordination
    # ------------------------------------------------------------------

    @staticmethod
    def _key(prefix: str, request_id: str) -> str:
        request_id = str(request_id).strip()
        if not request_id:
            raise MalformedSpecificationError("request_id must be a non-empty string")
        return f"{prefix}:{request_id}"

    def _shared_set(self, key: str, value: object, *, tags: Sequence[str] = ()) -> None:
        if not self.publish_shared_memory:
            return
        ttl = None if self.shared_memory_ttl_seconds <= 0 else self.shared_memory_ttl_seconds
        payload = to_json_safe(
            value,
            redact_sensitive=True,
            max_depth=12,
            max_items=max(self.max_shared_list_items, self.max_shared_model_entries),
            max_string_length=8_192,
        )
        try:
            self.shared_memory.set(
                key,
                payload,
                ttl=ttl,
                tags=list(tags),
                metadata={"agent": self.name, "agent_id": self.agent_id, "schema": "verification_agent.v1"},
            )
        except Exception as exc:
            self._handle_shared_memory_error("set", key, exc)

    def _handle_shared_memory_error(self, operation: str, key: str, exc: BaseException) -> None:
        if self.fail_on_shared_memory_error:
            raise BaseStateError(
                "VerificationAgent shared-memory operation failed",
                component=self.name,
                operation=f"shared_memory.{operation}",
                context={"key": key, "error_type": type(exc).__name__},
                cause=exc,
            ) from exc
        logger.warning(
            "VerificationAgent shared-memory %s degraded | key=%s | error=%s",
            operation,
            key,
            type(exc).__name__,
        )

    def _load_prior_references(self, request_ids: Sequence[str]) -> tuple[Mapping[str, object], ...]:
        if not self.publish_shared_memory or self.max_prior_references <= 0:
            return ()
        references: list[Mapping[str, object]] = []
        for request_id in tuple(request_ids)[: self.max_prior_references]:
            key = self._key(self.summary_key_prefix, request_id)
            try:
                value = self.shared_memory.get(key)
            except Exception as exc:
                self._handle_shared_memory_error("get", key, exc)
                continue
            if isinstance(value, Mapping):
                references.append(dict(value))
        return tuple(references)

    def _request_metadata(
        self,
        request: VerificationAgentRequest,
        prior_references: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        bounds = request.bounds
        assumptions = tuple(item.name for item in request.assumptions)
        payload: dict[str, object] = {
            "schema": "verification_agent.request.v1",
            "request_id": request.request_id,
            "operation": request.operation.value,
            "property_name": request.property_name or self._default_property_name(request),
            "artifact_type": type(request.artifact).__name__,
            "formal_property_type": type(request.formal_property).__name__ if request.formal_property is not None else None,
            "assumptions": assumptions[: self.max_shared_list_items],
            "backend_requested": request.backend_name,
            "bounds": None
            if bounds is None
            else {
                "max_states": bounds.max_states,
                "max_transitions": bounds.max_transitions,
                "max_depth": bounds.max_depth,
                "timeout_seconds": bounds.timeout_seconds,
            },
            "prior_request_ids": request.prior_request_ids[: self.max_prior_references],
            "prior_reference_count": len(prior_references),
            "metadata": dict(request.metadata),
        }
        payload["fingerprint"] = artifact_fingerprint(payload)
        return payload

    def _publish_outcome(self, request: VerificationAgentRequest, outcome: VerificationAgentOutcome) -> None:
        if not self.publish_shared_memory:
            return
        shared_result = self._shared_result_payload(outcome)
        status = outcome.verification_status
        summary: dict[str, object] = {
            "schema": "verification_agent.summary.v1",
            "request_id": request.request_id,
            "operation": request.operation.value,
            "execution_success": True,
            "verification_status": status.value if status is not None else None,
            "result_kind": self._non_status_label(outcome.result),
            "property_name": request.property_name or self._default_property_name(request),
            "evidence_available": self._evidence_available(outcome.result),
            "fingerprint": shared_result.get("fingerprint"),
        }
        if isinstance(outcome.result, VerificationResult):
            summary.update(
                {
                    "method": outcome.result.provenance.method.value,
                    "scope": outcome.result.provenance.scope.value,
                    "backend": outcome.result.provenance.backend,
                    "unknown_reason": outcome.result.unknown_reason,
                    "trace_available": outcome.result.trace is not None,
                    "proof_available": outcome.result.proof is not None,
                }
            )

        self._shared_set(
            self._key(self.result_key_prefix, request.request_id),
            shared_result,
            tags=("verification", "result", request.operation.value),
        )
        self._shared_set(
            self._key(self.summary_key_prefix, request.request_id),
            summary,
            tags=("verification", "summary", request.operation.value),
        )
        self._shared_set(
            self.latest_result_key,
            summary,
            tags=("verification", "latest"),
        )
        self._publish_event(
            "completed",
            {
                "request_id": request.request_id,
                "operation": request.operation.value,
                "verification_status": status.value if status is not None else None,
                "result_kind": self._non_status_label(outcome.result),
            },
        )

    def _publish_error(self, request: VerificationAgentRequest, error: BaseException) -> None:
        payload: dict[str, object] = {
            "schema": "verification_agent.error.v1",
            "request_id": request.request_id,
            "operation": request.operation.value,
            "execution_success": False,
            "error_type": type(error).__name__,
            "message": str(error)[:1000],
        }
        if isinstance(error, VerificationError):
            to_dict = getattr(error, "to_dict", None)
            if callable(to_dict):
                try:
                    serialized = to_dict()
                except Exception:
                    serialized = None
                if isinstance(serialized, Mapping):
                    payload["error"] = dict(serialized)
        self._shared_set(
            self._key(self.error_key_prefix, request.request_id),
            payload,
            tags=("verification", "error", request.operation.value),
        )
        self._publish_event(
            "failed",
            {
                "request_id": request.request_id,
                "operation": request.operation.value,
                "error_type": type(error).__name__,
            },
        )

    def _publish_event(self, event_type: str, payload: Mapping[str, object]) -> None:
        if not self.publish_shared_memory:
            return
        event = {
            "schema": "verification_agent.event.v1",
            "event_type": str(event_type),
            "agent": self.name,
            **dict(payload),
        }
        request_id = str(payload.get("request_id") or self.agent_id)
        event_key = f"{self._key(self.event_key_prefix, request_id)}:{str(event_type).strip()}"
        self._shared_set(
            event_key,
            event,
            tags=("verification", "event", str(event_type)),
        )

    def _shared_result_payload(self, outcome: VerificationAgentOutcome) -> dict[str, object]:
        full = outcome.to_dict()
        formal = full.get("formal_result")
        if isinstance(formal, Mapping):
            formal = dict(formal)
            model = formal.get("model")
            if isinstance(model, Mapping):
                items = list(model.items())
                formal["model"] = dict(items[: self.max_shared_model_entries])
                if len(items) > self.max_shared_model_entries:
                    formal["model_truncated"] = len(items) - self.max_shared_model_entries
            for key in ("assumptions", "limitations", "unsat_core"):
                value = formal.get(key)
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                    seq = list(value)
                    formal[key] = seq[: self.max_shared_list_items]
                    if len(seq) > self.max_shared_list_items:
                        formal[f"{key}_truncated"] = len(seq) - self.max_shared_list_items
            trace = formal.get("trace")
            if isinstance(trace, Mapping):
                trace = dict(trace)
                steps = trace.get("steps")
                if isinstance(steps, Sequence) and not isinstance(steps, (str, bytes, bytearray)):
                    step_list = list(steps)
                    trace["steps"] = step_list[: self.max_shared_trace_steps]
                    if len(step_list) > self.max_shared_trace_steps:
                        trace["steps_truncated"] = len(step_list) - self.max_shared_trace_steps
                formal["trace"] = trace
            if not self.publish_evidence_metadata:
                formal.pop("model", None)
                formal.pop("unsat_core", None)
                formal.pop("trace", None)
                formal.pop("proof", None)
            full["formal_result"] = formal
        full["prior_references"] = list(outcome.prior_references)[: self.max_prior_references]
        full["fingerprint"] = self._result_fingerprint(outcome.result)
        return full

    @staticmethod
    def _result_fingerprint(result: FormalResult) -> str:
        if isinstance(result, VerificationResult):
            return result_fingerprint(result)
        if isinstance(result, FixpointResult):
            payload = {
                "kind": "abstract_fixpoint",
                "value": result.value,
                "converged": result.converged,
                "iterations": result.iterations,
                "used_widening": result.used_widening,
                "reason": result.reason,
            }
        else:
            payload = {
                "kind": "abstract_invariant",
                "candidate": result.candidate,
                "post_fixpoint": result.post_fixpoint,
                "initial_included": result.initial_included,
                "inductive": result.inductive,
            }
        # to_json_safe turns arbitrary abstract-domain values into deterministic,
        # bounded data before the subsystem's canonical fingerprint helper is used.
        normalized = to_json_safe(payload, redact_sensitive=False, max_depth=8, max_items=1_000, max_string_length=8_192)
        if not isinstance(normalized, Mapping):
            normalized = {"payload": normalized}
        return artifact_fingerprint(dict(normalized))

    @staticmethod
    def _non_status_label(result: FormalResult) -> str:
        if isinstance(result, FixpointResult):
            return "abstract_fixpoint"
        if isinstance(result, AbstractInvariantCheck):
            return "abstract_invariant"
        return "verification_result"

    @staticmethod
    def _evidence_available(result: FormalResult) -> bool:
        if isinstance(result, VerificationResult):
            return bool(result.trace is not None or result.proof is not None or result.model or result.unsat_core)
        if isinstance(result, FixpointResult):
            return True
        return True

    # ------------------------------------------------------------------
    # BaseAgent metrics integration
    # ------------------------------------------------------------------

    def extract_performance_metrics(self, result: object) -> dict[str, float]:
        """Expose orchestration/formal-result telemetry without scoring correctness."""

        if not isinstance(result, Mapping):
            return {}
        formal = result.get("formal_result")
        if not isinstance(formal, Mapping):
            return {"verification.execution_success": 1.0 if result.get("execution_success") is True else 0.0}
        status = str(result.get("verification_status") or "")
        provenance = formal.get("provenance")
        elapsed = 0.0
        if isinstance(provenance, Mapping):
            try:
                elapsed = max(0.0, float(provenance.get("elapsed_seconds", 0.0) or 0.0))
            except (TypeError, ValueError, OverflowError):
                elapsed = 0.0
        return {
            "verification.execution_success": 1.0 if result.get("execution_success") is True else 0.0,
            "verification.formal_conclusive": 1.0 if status in {"verified", "refuted", "satisfiable", "unsatisfiable"} else 0.0,
            "verification.formal_unknown": 1.0 if status == "unknown" else 0.0,
            "verification.formal_bounded": 1.0 if status == "bounded" else 0.0,
            "verification.evidence_present": 1.0 if any(formal.get(key) for key in ("trace", "proof", "model", "unsat_core")) else 0.0,
            "verification.elapsed_seconds": elapsed,
        }

__all__ = [
    "VerificationAgent",
    "VerificationAgentOutcome",
    "VerificationAgentRequest",
    "VerificationOperation",
]


if __name__ == "__main__":
    print("\n=== Running Verification Agent ===\n")
    printer.status("TEST", "Verification Agent initialized", "info")
    from .agent_factory import AgentFactory
    from .collaborative.shared_memory import SharedMemory

    shared_memory = SharedMemory()
    agent_factory = AgentFactory()
    config = {"publish_shared_memory": False}

    agent = VerificationAgent(shared_memory=shared_memory, agent_factory=agent_factory, config=config)
    printer.status("START", agent, "info")


    system = TransitionSystem(
        states=(0, 1),
        initial_states=(0,),
        transitions=(
            # Labels are diagnostic only; model checking uses the transition relation.
            Transition(0, 1, "step"),
            Transition(1, 1, "stay"),
        ),
    )
    invariant = Invariant("non_negative", lambda state: cast(int, state) >= 0)
    smoke = agent.perform_task(
        {
            "request_id": "verification-agent-smoke",
            "operation": "invariant",
            "artifact": system,
            "formal_property": invariant,
        }
    )
    printer.status("TASK #1", smoke, "success" if smoke == "success" else "error")

    print("\n=== Test ran successfully ===\n")